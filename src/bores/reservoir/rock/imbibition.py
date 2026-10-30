import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.reservoir.rock.model import load_cell_array
from bores.types import CellArray, UnitSystem
from bores.units import UnitConversionTable, get_conversion_factors, resolve_unit_conversion_table
from bores.utils import scale

__all__ = ["IMBIBITION_ENDPOINT_KEYWORDS", "ImbibitionEndpoints"]

SATURATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("ISWL", "connate_water_saturation"),
    ("ISWCR", "critical_water_saturation"),
    ("ISWU", "maximum_water_saturation"),
    ("ISGL", "connate_gas_saturation"),
    ("ISGCR", "critical_gas_saturation"),
    ("ISGU", "maximum_gas_saturation"),
    ("ISOWCR", "residual_oil_saturation_water"),
    ("ISOGCR", "residual_oil_saturation_gas"),
    ("ISWLPC", "capillary_connate_water_saturation"),
    ("ISGLPC", "capillary_connate_gas_saturation"),
)
RELATIVE_PERMEABILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("IKRW", "maximum_water_relative_permeability"),
    ("IKRWR", "water_relative_permeability_at_residual_oil"),
    ("IKRO", "maximum_oil_relative_permeability"),
    ("IKRORW", "oil_relative_permeability_at_critical_water"),
    ("IKRORG", "oil_relative_permeability_at_critical_gas"),
    ("IKRG", "maximum_gas_relative_permeability"),
    ("IKRGR", "gas_relative_permeability_at_residual_oil"),
)
PRESSURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("IPCW", "maximum_oil_water_capillary_pressure"),
    ("IPCG", "maximum_gas_oil_capillary_pressure"),
)
IMBIBITION_ENDPOINT_KEYWORDS: tuple[str, ...] = tuple(
    keyword for keyword, _ in (*SATURATION_FIELDS, *RELATIVE_PERMEABILITY_FIELDS, *PRESSURE_FIELDS)
)
"""Every deck keyword that sets a per-cell imbibition end-point."""


@attrs.frozen(slots=True)
class ImbibitionEndpoints:
    """
    Per-cell end-points of the imbibition curves, from the deck's `I*` keywords.

    Each array is `None` when the deck does not give that keyword. Inside a given array a `NaN`
    cell was not assigned by any record, so that cell takes the end-point of its imbibition
    table, as in Eclipse. Saturations and relative permeabilities are dimensionless. The two
    capillary pressures are in `unit_system`.
    """

    connate_water_saturation: CellArray | None = None
    """`ISWL`: connate water saturation of the imbibition curves."""

    critical_water_saturation: CellArray | None = None
    """`ISWCR`: critical water saturation of the imbibition curves."""

    maximum_water_saturation: CellArray | None = None
    """`ISWU`: maximum water saturation of the imbibition curves."""

    connate_gas_saturation: CellArray | None = None
    """`ISGL`: connate gas saturation of the imbibition curves."""

    critical_gas_saturation: CellArray | None = None
    """`ISGCR`: critical gas saturation of the imbibition curves."""

    maximum_gas_saturation: CellArray | None = None
    """`ISGU`: maximum gas saturation of the imbibition curves."""

    residual_oil_saturation_water: CellArray | None = None
    """`ISOWCR`: critical oil-in-water saturation of the imbibition curves."""

    residual_oil_saturation_gas: CellArray | None = None
    """`ISOGCR`: critical oil-in-gas saturation of the imbibition curves."""

    capillary_connate_water_saturation: CellArray | None = None
    """`ISWLPC`: connate water saturation that scales only the imbibition capillary pressure."""

    capillary_connate_gas_saturation: CellArray | None = None
    """`ISGLPC`: connate gas saturation that scales only the imbibition capillary pressure."""

    maximum_water_relative_permeability: CellArray | None = None
    """`IKRW`: maximum water relative permeability of the imbibition curve."""

    water_relative_permeability_at_residual_oil: CellArray | None = None
    """`IKRWR`: water relative permeability at the critical oil-in-water saturation."""

    maximum_oil_relative_permeability: CellArray | None = None
    """`IKRO`: maximum oil relative permeability of the imbibition curves."""

    oil_relative_permeability_at_critical_water: CellArray | None = None
    """`IKRORW`: oil relative permeability at the critical water saturation."""

    oil_relative_permeability_at_critical_gas: CellArray | None = None
    """`IKRORG`: oil relative permeability at the critical gas saturation."""

    maximum_gas_relative_permeability: CellArray | None = None
    """`IKRG`: maximum gas relative permeability of the imbibition curve."""

    gas_relative_permeability_at_residual_oil: CellArray | None = None
    """`IKRGR`: gas relative permeability at the critical oil-in-gas saturation."""

    maximum_oil_water_capillary_pressure: CellArray | None = None
    """`IPCW`: maximum oil-water capillary pressure of the imbibition curve, in `unit_system`."""

    maximum_gas_oil_capillary_pressure: CellArray | None = None
    """`IPCG`: maximum gas-oil capillary pressure of the imbibition curve, in `unit_system`."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system of the two capillary pressure arrays."""

    def __attrs_post_init__(self) -> None:
        for keyword, field in SATURATION_FIELDS + RELATIVE_PERMEABILITY_FIELDS:
            array = getattr(self, field)
            if array is not None and (
                np.nanmin(array, initial=0.0) < 0.0 or np.nanmax(array, initial=0.0) > 1.0
            ):
                raise ValidationError(f"`{keyword}` must lie between 0 and 1 in every cell.")
        for keyword, field in PRESSURE_FIELDS:
            array = getattr(self, field)
            if array is not None and np.nanmin(array, initial=0.0) < 0.0:
                raise ValidationError(f"`{keyword}` must not be negative in any cell.")

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return the end-points in `target` units.

        Only the capillary pressures change, because saturations and relative permeabilities are
        dimensionless.

        :param target: Desired `UnitSystem`.
        :param table: Optional custom conversion table; `None` uses the default.
        :returns: New `ImbibitionEndpoints` in `target` units.
        """
        if target == self.unit_system:
            return self
        factor = get_conversion_factors(
            self.unit_system, target, table=resolve_unit_conversion_table(table)
        )["pressure"]
        return attrs.evolve(
            self,
            maximum_oil_water_capillary_pressure=scale(
                self.maximum_oil_water_capillary_pressure, factor
            ),
            maximum_gas_oil_capillary_pressure=scale(
                self.maximum_gas_oil_capillary_pressure, factor
            ),
            unit_system=target,
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        n_cells: int,
        dtype: npt.DTypeLike = None,
    ) -> Self | None:
        """
        Load the imbibition end-points from a deck, in the deck's unit system.

        :param deck_file: The deck to read.
        :param n_cells: Number of cells in the grid.
        :param dtype: Floating point type of the arrays, the library default if not given.
        :returns: The end-points, or `None` when the deck has none of the `I*` end-point
            keywords.
        :raises ValidationError: If an array does not have one value per cell, or a value is out
            of range.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        arrays: dict[str, CellArray] = {}
        for keyword, field in (
            *SATURATION_FIELDS,
            *RELATIVE_PERMEABILITY_FIELDS,
            *PRESSURE_FIELDS,
        ):
            array = load_cell_array(deck_file, keyword, n_cells, dtype=dtype)
            if array is not None:
                arrays[field] = array
        if not arrays:
            return None
        return cls(**arrays, unit_system=deck_file.unit_system)
