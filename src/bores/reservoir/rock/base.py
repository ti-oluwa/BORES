import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.regions import load_region_array

if typing.TYPE_CHECKING:
    from bores.blackoil.satfunc.regions import SatFunc
from bores.reservoir.rock.compressibility import (
    RockCompressibility,
    RockCompressibilityTables,
)
from bores.serde.stores import StoreSerializable
from bores.types import (
    CellArray,
    IntCellArray,
    InterpolationMethod,
    Number,
    UnitSystem,
)
from bores.utils import scale

__all__ = ["Permeability", "Rock"]


def load_cell_array(
    deck_file: DeckFile, keyword: str, n_cells: int, dtype: npt.DTypeLike = None
) -> CellArray | None:
    array = deck_file.get(keyword)
    if array is None:
        return None

    array = array.astype(dtype, copy=False)
    if array.size != n_cells:
        raise ValidationError(f"{keyword} has {array.size} values; expected {n_cells}.")
    return typing.cast(CellArray, array)


@attrs.frozen(slots=True)
class Permeability(StoreSerializable):
    """
    Absolute permeability tensor stored as three orthogonal components.

    If only `x` is supplied, `y` and `z` default to `x` (isotropic
    assumption). The geometric-mean `mean` is computed automatically when
    not provided.

    Units should follow the parent `Rock.unit_system`.
    """

    x: CellArray
    """
    Shape (n_cells,) - permeability in the x-direction.

    Units: mD (FIELD / METRIC / LAB) or m² (SI).
    Must be strictly positive for every active cell.
    """

    y: CellArray = attrs.field(factory=lambda: np.empty(0))
    """
    Shape (n_cells,) - permeability in the y-direction.

    Defaults to `x` (isotropic y) when not supplied.
    Units: same as `x`.
    """

    z: CellArray = attrs.field(factory=lambda: np.empty(0))
    """
    Shape (n_cells,) - permeability in the z-direction.

    Defaults to `x` (isotropic z) when not supplied.
    Units: same as `x`.
    """

    mean: CellArray = attrs.field(factory=lambda: np.empty(0))
    """
    Shape (n_cells,) - geometric-mean permeability (Kx·Ky·Kz)^(1/3).

    Computed automatically from `x`, `y`, `z` when not supplied.
    Units: same as `x`.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all quantities on this object are expressed.
    """

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)
        if self.mean.size == 0:
            if np.array_equal(self.x, self.y) and np.array_equal(self.x, self.z):
                object.__setattr__(self, "mean", self.x)
            else:
                object.__setattr__(self, "mean", (self.x * self.y * self.z) ** (1.0 / 3.0))

    def scale(self, factor: Number) -> Self:
        """Return a new instance with all components multiplied by *factor*."""
        if factor == 1:
            return self
        return attrs.evolve(
            self,
            x=scale(self.x, factor),
            y=scale(self.y, factor),
            z=scale(self.z, factor),
            mean=scale(self.mean, factor),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
        factor: Number | None = None,
    ) -> Self:
        """
        Return a new `Permeability` with all quantities rescaled
        to *target*.

        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `Permeability` in *target* units.
        """
        if target == self.unit_system:
            return self

        if factor is None:
            factors = get_conversion_factors(self.unit_system, target, table=table)
            factor = factors["permeability"]
        if factor == 1:
            return self

        return attrs.evolve(
            self,
            x=scale(self.x, factor),
            y=scale(self.y, factor),
            z=scale(self.z, factor),
            mean=scale(self.mean, factor),
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        n_cells: int,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        unit_system = deck_file.unit_system
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        permx = load_cell_array(deck_file, "PERMX", n_cells, dtype=dtype)
        if permx is None:
            raise ValidationError("`PERMX` is required but not found in the DeckFile.")

        permy = load_cell_array(deck_file, "PERMY", n_cells, dtype=dtype)
        # We do not copy here to save memory, since permeability is a static prop through out any simulation
        # Except the user then tries to modify it. Then its up to them as its already indicated that the class'
        # attributes are immutable and should be treated as such.
        if permy is None:
            permy = permx
        permz = load_cell_array(deck_file, "PERMZ", n_cells, dtype=dtype)
        if permz is None:
            permz = permx
        return cls(x=permx, y=permy, z=permz, unit_system=unit_system)


@attrs.frozen(slots=True)
class Rock(StoreSerializable):
    """
    Static petrophysical properties of the reservoir rock.

    These arrays are constant between simulation time steps and are populated
    from deck keywords such as `PORO`, `PERMX/Y/Z`, `NTG`, `SWCON`,
    `SWCRIT`, and `TEMPVD`.

    All saturation arrays are dimensionless fractions in [0, 1].
    Use `convert(target)` to rescale dimensional quantities to another
    unit system.
    """

    porosity: CellArray
    """
    Shape (n_cells,) - pore volume fraction (dimensionless, [0, 1]).

    Used to compute pore volume: PV = φ x NTG x V_cell.
    """

    absolute_permeability: Permeability
    """
    Absolute permeability tensor.

    Units: mD (FIELD / METRIC / LAB), m² (SI).
    """

    net_to_gross: CellArray
    """
    Shape (n_cells,) - net-to-gross ratio (dimensionless, [0, 1]).

    Fraction of the gross cell volume that is net reservoir rock.
    Applied as a multiplier in pore-volume and transmissibility calculations.
    """

    connate_water_saturation: CellArray
    """
    Shape (n_cells,) - connate (initial irreducible) water saturation
    (fraction).

    Lower bound on water saturation; set from geological initial conditions.
    """

    irreducible_water_saturation: CellArray
    """
    Shape (n_cells,) - irreducible water saturation during imbibition
    (fraction).

    Equal to or greater than `connate_water_saturation`.
    """

    residual_oil_saturation_water_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of water flooding
    (Sor,w - fraction).

    Oil is immobile below this saturation during water-flood imbibition.
    """

    residual_oil_saturation_gas_flood: CellArray
    """
    Shape (n_cells,) - residual oil saturation at end of gas flooding
    (Sor,g - fraction).

    Oil is immobile below this saturation during gas injection.
    """

    residual_gas_saturation: CellArray
    """
    Shape (n_cells,) - residual gas saturation during imbibition (fraction).

    Gas is immobile below this saturation when water or liquid displaces gas.
    """

    compressibility: RockCompressibilityTables | None = None
    """
    Formation compressibility tables - one per `ROCKNUM` region in `unit_system`.

    Use `get_compressibility(...)` to get a `RockCompressibility` for given pressures
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """
    Unit system in which all dimensional quantities on this object are expressed.

    Dimensionless arrays (porosity, NTG, saturations) are unaffected by
    unit conversion.
    """

    def __attrs_post_init__(self) -> None:
        if (
            self.compressibility is not None
            and self.compressibility.unit_system != self.unit_system
        ):
            raise ValidationError(
                "`compressibility.unit_system does` not match `unit_system`: "
                f"{self.compressibility.unit_system} != {self.unit_system}"
            )

    def get_compressibility(
        self,
        *,
        pressure: CellArray,
        rock_region: IntCellArray | None = None,
        unit_system: UnitSystem | None = None,
        dtype: npt.DTypeLike = None,
    ) -> RockCompressibility:
        """
        Return formation compressibility tensor.

        Units: 1/psi (FIELD), 1/bar (METRIC), 1/atm (LAB), 1/Pa (SI).
        """
        target_unit_system = unit_system if unit_system is not None else self.unit_system
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        if self.compressibility is None:
            return RockCompressibility(
                reference_pressure=pressure,
                compressibility=typing.cast(CellArray, np.zeros_like(pressure, dtype=dtype)),
                unit_system=target_unit_system,
            )

        compressibility = self.compressibility.get_compressibility(
            pressure=pressure,
            rock_region=rock_region,
            unit_system=target_unit_system,
            dtype=dtype,
        )
        return compressibility

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `Rock` with all dimensional quantities rescaled
        to *target*.

        Dimensionless arrays (porosity, NTG, saturations) are copied unchanged.
        Conversion factors are sourced from `get_conversion_factors`.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `Rock` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            porosity=self.porosity,
            absolute_permeability=self.absolute_permeability.scale(factors["permeability"]),
            net_to_gross=self.net_to_gross,
            compressibility=self.compressibility.convert(target, table=table)
            if self.compressibility is not None
            else None,
            connate_water_saturation=self.connate_water_saturation,
            irreducible_water_saturation=self.irreducible_water_saturation,
            residual_oil_saturation_water_flood=self.residual_oil_saturation_water_flood,
            residual_oil_saturation_gas_flood=self.residual_oil_saturation_gas_flood,
            residual_gas_saturation=self.residual_gas_saturation,
            unit_system=target,
        )

    @staticmethod
    def get_saturation_endpoints(
        satfunc: "SatFunc",
        saturation_region: IntCellArray,
        n_cells: int,
        dtype: npt.DTypeLike,
    ) -> dict[str, CellArray]:
        """
        Derive per-cell saturation-endpoint fallbacks from each cell's
        SATNUM relative-permeability model, for whichever of the five
        endpoint keywords (`SWCON`, `SWCRIT`, `SOWCR`, `SOGCR`, `SGCR`)
        aren't present in the deck.

        Uses `RelativePermeabilityTable.get_saturation_endpoints()`.

        This is what a simulator would do automatically when
        explicit endpoint arrays are absent (rather than silently treating
        every cell as having zero connate water / zero residual
        saturations, as a large error whenever the reservoir engineer defined
        endpoints only through the relative-permeability model/tables, as
        many hand-built decks do).

        `irreducible_water_saturation` (`SWCRIT`) has no separate source
        and defaults to `connate_water_saturation`, matching what the rest
        of this codebase already assumes when the two coincide.

        :returns: Dictionary keyed by `Rock` field name, each shape `(n_cells,)`.
        """
        connate_water_saturation = np.zeros(n_cells, dtype=dtype)
        residual_oil_saturation_water_flood = np.zeros(n_cells, dtype=dtype)
        residual_oil_saturation_gas_flood = np.zeros(n_cells, dtype=dtype)
        residual_gas_saturation = np.zeros(n_cells, dtype=dtype)

        for satnum in np.unique(saturation_region):
            mask = saturation_region == satnum
            endpoints = satfunc.region(satnum).relative_permeability.get_saturation_endpoints()
            connate_water_saturation[mask] = endpoints.connate_water
            residual_oil_saturation_water_flood[mask] = endpoints.residual_oil_water
            residual_oil_saturation_gas_flood[mask] = endpoints.residual_oil_gas
            residual_gas_saturation[mask] = endpoints.residual_gas

        return {
            "connate_water_saturation": connate_water_saturation,
            "irreducible_water_saturation": connate_water_saturation.copy(),
            "residual_oil_saturation_water_flood": residual_oil_saturation_water_flood,
            "residual_oil_saturation_gas_flood": residual_oil_saturation_gas_flood,
            "residual_gas_saturation": residual_gas_saturation,
        }

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        grid: Grid,
        rock_region: IntCellArray | None = None,
        satfunc: "SatFunc | None" = None,
        saturation_region: IntCellArray | None = None,
        interpolation_method: InterpolationMethod = "linear",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `Rock` from a parsed `DeckFile` and an already-loaded `Grid`.

        Reads `PORO`, `PERMX/Y/Z`, `NTG`, `SWCON`, `SWCRIT`, `SOWCR`,
        `SOGCR`, `SGCR`, `ROCK`/`ROCKTAB`, and `TEMPVD`/`RTEMP`.

        `SWCON`/`SWCRIT`/`SOWCR`/`SOGCR`/`SGCR` are per-cell saturation
        *endpoint* arrays. This is relatively uncommon in hand-built decks, which
        usually define these endpoints implicitly through the saturation
        function tables (`SWOF`/`SGOF`) instead.

        Whichever of the five is absent from the deck is derived from
        `satfunc`'s per-SATNUM tables when `satfunc` is supplied (see
        `get_saturation_endpoints`); only truly defaults to `0.0`
        if `satfunc` isn't given either. Explicit deck arrays always take
        precedence, per keyword independently.

        :param deck_file: Parsed `DeckFile` containing PROPS/GRID keywords.
        :param grid: Already-loaded `Grid` (provides `n_cells` and cell
            centroid depths for temperature interpolation).
        :param satfunc: Optional `SatFunc`, used to derive any
            of the five saturation-endpoint arrays not explicitly present
            in `deck_file`, from each cell's SATNUM saturation-function
            table. Strongly recommended. Without it, any endpoint the deck
            doesn't supply explicitly silently defaults to `0.0` for every
            cell, which is rarely physically correct.
        :param saturation_region: Optional per-cell SATNUM array, used
            (only) for the `satfunc`-based derivation above. If omitted,
            loaded from the deck's own `SATNUM` keyword (defaulting to
            region 1 everywhere if that's absent too).
        :returns: `Rock` in the deck's unit system.
        :raises ValidationError: If required keywords (`PORO`, `PERMX`) are
            missing.
        """
        unit_system = deck_file.unit_system
        n_cells = grid.n_cells
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        def required(keyword: str) -> CellArray:
            data = load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if data is None:
                raise ValidationError(f"`{keyword}` is required but not found in the DeckFile.")
            return data

        def optional(keyword: str, default: float) -> CellArray:
            data = load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if data is None:
                return np.full(n_cells, default, dtype=dtype)
            return data

        def get_saturation_endpoint(
            keyword: str,
            field: str,
            table_derived: typing.Mapping[str, CellArray] | None,
        ) -> CellArray:
            data = load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if data is not None:
                return data
            if table_derived is not None:
                return table_derived[field]
            return np.zeros(n_cells, dtype=dtype)

        permeability = Permeability.from_deck(deck_file, n_cells=n_cells, dtype=dtype)
        compressibility: RockCompressibilityTables | None = None
        if deck_file.has("ROCK") or deck_file.has("ROCKTAB"):
            compressibility = RockCompressibilityTables.from_deck(
                deck_file, interpolation_method=interpolation_method, dtype=dtype
            )

        if rock_region is None:
            rock_region = load_region_array(deck_file, "ROCKNUM", n_cells)

        table_derived_endpoints: dict[str, CellArray] | None = None
        if satfunc is not None:
            if saturation_region is None:
                saturation_region = load_region_array(deck_file, "SATNUM", n_cells)
            if saturation_region is not None:
                table_derived_endpoints = cls.get_saturation_endpoints(
                    satfunc=satfunc,
                    saturation_region=saturation_region,
                    n_cells=n_cells,
                    dtype=dtype,
                )
            else:
                warnings.warn(
                    "`satfunc` was provided but rock saturation regions (`SATNUM`) could not be determined from deck"
                    "Pass `saturation_region` or exclude `satfunc` to silence this warning",
                    category=UserWarning,
                    stacklevel=3,
                )
        return cls(
            porosity=required("PORO"),
            absolute_permeability=permeability,
            net_to_gross=optional("NTG", 1.0),
            compressibility=compressibility,
            connate_water_saturation=get_saturation_endpoint(
                "SWCON", "connate_water_saturation", table_derived_endpoints
            ),
            irreducible_water_saturation=get_saturation_endpoint(
                "SWCRIT", "irreducible_water_saturation", table_derived_endpoints
            ),
            residual_oil_saturation_water_flood=get_saturation_endpoint(
                "SOWCR", "residual_oil_saturation_water_flood", table_derived_endpoints
            ),
            residual_oil_saturation_gas_flood=get_saturation_endpoint(
                "SOGCR", "residual_oil_saturation_gas_flood", table_derived_endpoints
            ),
            residual_gas_saturation=get_saturation_endpoint(
                "SGCR", "residual_gas_saturation", table_derived_endpoints
            ),
            unit_system=unit_system,
        )
