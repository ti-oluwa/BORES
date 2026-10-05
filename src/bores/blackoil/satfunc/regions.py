import logging
import typing

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.satfunc.capillary_pressure.tables import (
    ThreePhaseCapillaryPressureTable,
)
from bores.blackoil.satfunc.relperm.mixing_rules import MixingRule
from bores.blackoil.satfunc.relperm.tables import (
    MinimumRelPerm,
    ThreePhaseRelPermTable,
)
from bores.blackoil.satfunc.tables import SatFuncTables
from bores.constants import UnitConversionTable
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import Spacing, UnitSystem

logger = logging.getLogger(__name__)

__all__ = ["SatFunc"]


class SatFunc:
    """
    Multi-region saturation-function tables keyed by 1-based `SATNUM` region index.

    Eclipse supports multiple saturation function regions via the `SATNUM`
    keyword - each cell is assigned a region index and its relative
    permeability and capillary pressure are evaluated from the corresponding
    `SatFuncTables` instance.

    Use `region(satnum)` to retrieve the tables for a given region, and
    `from_deck` to construct from a deck.
    """

    __slots__ = ("tables", "unit_system")

    def __init__(
        self,
        tables: dict[int, SatFuncTables],
        *,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Build a `SatFunc` from a pre-built regional tables dict.

        :param tables: Mapping from 1-based SATNUM index to `SatFuncTables`.
        :param unit_system: Expected unit system for all tables. If omitted,
            it is inferred from the first table and every other table is
            required to match it.
        :raises ValidationError: If *tables* is empty, or if any table's
            unit system does not match *unit_system* (explicit or inferred).
        """
        if not tables:
            raise ValidationError("`tables` must contain at least one entry.")

        expected_unit_system = unit_system or next(iter(tables.values())).unit_system
        mismatched = {
            satnum: table.unit_system
            for satnum, table in tables.items()
            if table.unit_system != expected_unit_system
        }
        if mismatched:
            raise ValidationError(
                f"All tables must share `{expected_unit_system.value!r}` as "
                f"`{self.__class__.__name__}.unit_system`; mismatches "
                f"(satnum -> unit_system): "
                f"{ {k: v.value for k, v in mismatched.items()} }."
            )

        self.tables = tables
        self.unit_system = expected_unit_system

    @property
    def n_regions(self) -> int:
        """Number of saturation function regions."""
        return len(self.tables)

    def region(self, satnum: int) -> SatFuncTables:
        """
        Return the `SatFuncTables` for a given 1-based region index.

        :param satnum: 1-based SATNUM region index.
        :returns: `SatFuncTables` for that region.
        :raises KeyError: If the region index does not exist.
        """
        tables = self.tables.get(satnum)
        if tables is None:
            available = sorted(self.tables.keys())
            raise KeyError(
                f"Saturation function region {satnum} not found. Available regions: {available}."
            )
        return tables

    @classmethod
    def from_one(cls, tables: SatFuncTables) -> Self:
        """
        Wrap a single `SatFuncTables` as region 1.

        Convenience factory for the common single-region case.

        :param tables: `SatFuncTables` instance.
        :returns: `SatFunc` with one entry at key 1.
        """
        return cls(tables={1: tables})

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `SatFunc` with capillary pressure in every
        region rescaled to *target*.

        Relative permeability is dimensionless and is unaffected.

        :param target: Target `UnitSystem`.
        :returns: New `SatFunc` in *target* units.
        """
        return self.__class__(
            tables={
                satnum: tables.convert(target, table=table)
                for satnum, tables in self.tables.items()
            }
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        mixing_rule: MixingRule | str | None = None,
        keyword_family: typing.Literal["first", "second", "auto"] = "auto",
        number_of_base_points: int = 200,
        number_of_endpoint_extra_points: int = 30,
        spacing: Spacing = "cosine",
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        include_capillary_pressure: bool = True,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build all saturation function regions from a parsed `DeckFile`.

        Detects which Eclipse saturation-function keyword family is present
        (`SWOF`/`SGOF` or `SWFN`/`SGFN` + `SOF2`/`SOF3`) and builds one
        `SatFuncTables` per `SATNUM` region, each combining a
        `ThreePhaseRelPermTable` and (when available) a
        `ThreePhaseCapillaryPressureTable`.

        The number of regions is inferred from the longest region list across
        the relevant keywords (`SWOF`/`SWFN` for oil-water, `SGOF`/`SGFN` for
        gas-oil).

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param mixing_rule: Three-phase oil mixing rule (name or callable),
            applied to every region's relative permeability table.
        :param keyword_family: `"first"`, `"second"`, or `"auto"` (default).
        :param number_of_base_points: Passed to relperm PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to relperm/capillay pressure PCHIP endpoint
            enrichment.
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param min_wetting_relperm: Min-value sentinel for wetting phase kr in
            every region's relperm sub-tables.
        :param min_non_wetting_relperm: Min-value sentinel for non-wetting phase
            kr in every region's relperm sub-tables.
        :param include_capillary_pressure: When `True` (default), also build
            the capillary pressure table for each region. When `False`, skip
            capillary pressure entirely (useful for runs that ignore Pc).
        :param dtype: Array dtype shared by every region's tables.
        :returns: `SatFunc` keyed by 1-based `SATNUM` index.
        :raises ValidationError: If no recognised saturation-function keywords
            are found.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()

        def get_region_count(keyword: str) -> int:
            all_regions = deck_file.get(keyword)
            return len(all_regions) if all_regions is not None else 0

        n_regions = max(
            get_region_count("SWOF"),
            get_region_count("SGOF"),
            get_region_count("SWFN"),
            get_region_count("SGFN"),
        )
        if n_regions == 0:
            raise ValidationError(
                "No recognised saturation-function keywords found in the `DeckFile`. "
                "Expected one of: `SWOF`, `SGOF` (first family) or `SWFN`, `SGFN` "
                "(second family)."
            )

        tables: dict[int, SatFuncTables] = {}
        for region_index in range(n_regions):
            satnum = region_index + 1  # 1-based

            relative_permeability = ThreePhaseRelPermTable.from_deck(
                deck_file=deck_file,
                satnum=satnum,
                mixing_rule=mixing_rule,
                keyword_family=keyword_family,
                number_of_base_points=number_of_base_points,
                number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                spacing=spacing,
                min_wetting_relperm=min_wetting_relperm,
                min_non_wetting_relperm=min_non_wetting_relperm,
                dtype=dtype,
            )

            capillary_pressure: ThreePhaseCapillaryPressureTable | None = None
            if include_capillary_pressure:
                capillary_pressure = ThreePhaseCapillaryPressureTable.from_deck(
                    deck_file=deck_file,
                    satnum=satnum,
                    keyword_family=keyword_family,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                    spacing=spacing,
                    dtype=dtype,
                )

            tables[satnum] = SatFuncTables(
                relative_permeability=relative_permeability, capillary_pressure=capillary_pressure
            )
            logger.debug(
                "Built rock-fluid tables for `SATNUM` region %d (capillary_pressure=%s)",
                satnum,
                include_capillary_pressure,
            )
        return cls(tables=tables)

    def __getitem__(self, key: int) -> SatFuncTables:
        return self.region(key)

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def __contains__(self, key: object) -> bool:
        return key in self.tables
