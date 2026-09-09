"""Entry point for parsing Eclipse / GRDECL decks."""

import typing
from collections.abc import Collection

from typing_extensions import TypeVar

from bores.datastructures import GridDimensions
from bores.deck.core import (
    Deck,
    DeckParseError,
    resolve_source,
    strip_comments,
)
from bores.deck.keywords.base import Keyword, get_schedule_times
from bores.deck.keywords.grid import (
    ACTNUM,
    COORD,
    DIMENS,
    DX,
    DXV,
    DY,
    DYV,
    DZ,
    DZV,
    FAULTS,
    GRIDUNIT,
    MAPAXES,
    MAPUNITS,
    MULTFLT,
    MULTX,
    MULTX_MINUS,
    MULTY,
    MULTY_MINUS,
    MULTZ,
    MULTZ_MINUS,
    NNC,
    NTG,
    PERMX,
    PERMY,
    PERMZ,
    PINCH,
    PINCHOUT,
    PORO,
    PORV,
    SPECGRID,
    TOPS,
    ZCORN,
)
from bores.deck.keywords.props import (
    DENSITY,
    PVCO,
    PVDG,
    PVDO,
    PVTG,
    PVTO,
    PVTW,
    ROCK,
    ROCKTAB,
    SGFN,
    SGOF,
    SOF2,
    SOF3,
    SWFN,
    SWOF,
)
from bores.deck.keywords.regions import EQLNUM, FIPNUM, IMBNUM, PVTNUM, ROCKNUM, SATNUM
from bores.deck.keywords.runspec import (
    DISGAS,
    EQLDIMS,
    FIELD,
    GAS,
    LAB,
    METRIC,
    NOSIM,
    OIL,
    REGDIMS,
    START,
    TABDIMS,
    TITLE,
    UNIFIN,
    UNIFOUT,
    VAPOIL,
    WATER,
    WELLDIMS,
)
from bores.deck.keywords.schedule import (
    COMPDAT,
    DATES,
    GCONINJE,
    GCONPROD,
    GECON,
    GRUPTREE,
    TSTEP,
    WCONHIST,
    WCONINJE,
    WCONINJH,
    WCONPROD,
    WDFAC,
    WECON,
    WELOPEN,
    WELPI,
    WELSPECS,
    WELTARG,
    WGRUPCON,
    WPAVE,
    WPIMULT,
    WTEST,
)
from bores.deck.keywords.solution import (
    EQUIL,
    PRESSURE,
    RESTART,
    RS,
    RSVD,
    RTEMP,
    RTEMPVD,
    RV,
    RVVD,
    SGAS,
    SOIL,
    SWAT,
    TEMPVD,
)
from bores.deck.keywords.summary import (
    FGPR,
    FGPT,
    FOPR,
    FOPT,
    FWPR,
    FWPT,
    RGIP,
    ROIP,
    RPTRST,
    RPTSCHED,
    RWIP,
    WBHP,
    WGPR,
    WOPR,
    WTHP,
    WWPR,
)
from bores.deck.operators import Operation, resolve_operations
from bores.types import TextOrPath, UnitSystem

__all__ = ["DataFile", "DeckFile"]

T = TypeVar("T")

DIMENSION_KEYWORDS = ("SPECGRID", "DIMENS")

DEFAULT_KEYWORDS = frozenset({
    # RUNSPEC ###
    DISGAS,
    EQLDIMS,
    FIELD,
    GAS,
    LAB,
    METRIC,
    NOSIM,
    OIL,
    REGDIMS,
    START,
    TABDIMS,
    TITLE,
    UNIFIN,
    UNIFOUT,
    VAPOIL,
    WATER,
    WELLDIMS,
    # GRID ###
    ACTNUM,
    COORD,
    DIMENS,
    DX,
    DXV,
    DY,
    DYV,
    DZ,
    DZV,
    FAULTS,
    GRIDUNIT,
    MAPAXES,
    MAPUNITS,
    MULTFLT,
    MULTX,
    MULTX_MINUS,
    MULTY,
    MULTY_MINUS,
    MULTZ,
    MULTZ_MINUS,
    NNC,
    NTG,
    PERMX,
    PERMY,
    PERMZ,
    PINCH,
    PINCHOUT,
    PORO,
    PORV,
    SPECGRID,
    TOPS,
    ZCORN,
    # REGIONS ###
    EQLNUM,
    FIPNUM,
    IMBNUM,
    PVTNUM,
    ROCKNUM,
    SATNUM,
    # PROPS ###
    DENSITY,
    PVCO,
    PVDG,
    PVDO,
    PVTG,
    PVTO,
    PVTW,
    ROCK,
    ROCKTAB,
    SGFN,
    SGOF,
    SOF2,
    SOF3,
    SWFN,
    SWOF,
    # SOLUTION ###
    EQUIL,
    PRESSURE,
    RESTART,
    RS,
    RV,
    SGAS,
    SOIL,
    SWAT,
    RTEMP,
    RTEMPVD,
    TEMPVD,
    RSVD,
    RVVD,
    # SCHEDULE ###
    COMPDAT,
    DATES,
    GCONINJE,
    GCONPROD,
    GRUPTREE,
    TSTEP,
    WECON,
    WELOPEN,
    WELSPECS,
    WELTARG,
    WCONINJE,
    WCONPROD,
    WPIMULT,
    WTEST,
    WGRUPCON,
    GECON,
    WELPI,
    WPAVE,
    WCONHIST,
    WCONINJH,
    WDFAC,
    # SUMMARY ###
    FGPR,
    FGPT,
    FOPR,
    FOPT,
    FWPR,
    FWPT,
    RGIP,
    ROIP,
    RPTRST,
    RPTSCHED,
    RWIP,
    WBHP,
    WGPR,
    WOPR,
    WTHP,
    WWPR,
})


class DeckFile:
    """
    A parsed Eclipse deck file: comment-stripped, `INCLUDE`-resolved text plus a
    registry of `bores.deck.keywords.base.Keyword` objects, exposed through `get`.

    **Usage**:

    ```python
    from bores.deck import DeckFile, DEFAULT_KEYWORDS
    from bores.deck import keywords as k

    df = DeckFile("path/to/model.DATA")
    poro = df.get("PORO")  # ndarray (n_cells,) or None (string - poor typing support)
    faults = df.get(k.FAULTS)  # List[Dict] or None (this - better typing support)

    # With extra keywords not in the default set:
    df2 = DeckFile(text, keywords=DEFAULT_KEYWORDS | [MyCustomKeyword()])
    val = df2.get("MYCUSTOM")
    ```

    **Adding a new keyword**:

    ```python
    from bores.deck.keywords.base import ArrayKeyword

    df.add_keywords(ArrayKeyword("MYARRAY"))
    val = df.get("MYARRAY")
    ```

    **Or pass it at construction time**:
    ```python
    df = DeckFile(source, keywords=[ArrayKeyword("MYARRAY"), ...])
    ```
    """

    __slots__ = (
        "_cache",
        "_deck",
        "_operation_targets",
        "_operations",
        "_registry",
        "_schedule_times",
        "dimensions",
        "unit_system",
    )

    def __init__(
        self,
        source: TextOrPath,
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        encoding: str = "ascii",
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        :param source: A file path, raw deck text `str`, or raw deck
            `bytes`. Filesystem paths and `Path` objects are read from
            disk; `INCLUDE` directives are resolved relative to the source
            file's directory. Raw text input drops `INCLUDE` directives with a warning.
        :param keywords: Supported `bores.deck.keywords.base.Keyword`
            instances. Only supported keywords can be read from the file.
        :param encoding: Text encoding for file/bytes input (default `"ascii"`).
        :param unit_system: The unit system to use for the deck. If not provided,
            the deck is scanned for a `FIELD`, `METRIC`, or `LAB` keyword.
            If none is found, a `DeckParseError` is raised.
            Do not provide this argument if you want the unit system to be auto-detected.
        :raises DeckParseError: If a file cannot be read.
        """
        text = resolve_source(source, encoding=encoding)
        clean_text = strip_comments(text)
        self._deck = Deck(clean_text)

        self._registry: dict[str, Keyword[typing.Any]] = {
            keyword.name: keyword for keyword in keywords
        }
        self._cache: dict[str, typing.Any] = {}
        self.dimensions: GridDimensions | None = self._resolve_dimensions()
        deck_unit_system = unit_system or self._resolve_unit_system()
        if deck_unit_system is None:
            raise DeckParseError(
                "Could not detect deck unit system. Perhaps you passed a standalone PVT, GRDECL or relperm include file. "
                "Provide `unit_system` on instantiation."
            )

        self.unit_system: UnitSystem = deck_unit_system
        self._operations: list[Operation] | None = (
            resolve_operations(self._deck, self.dimensions)
            if self.dimensions is not None
            else None
        )
        self._operation_targets: frozenset[str] = (
            frozenset(operation.target for operation in self._operations)
            if self._operations
            else frozenset()
        )

        time_unit = "days"
        if deck_unit_system == UnitSystem.LAB:
            time_unit = "hours"
        elif deck_unit_system == UnitSystem.SI:
            time_unit = "seconds"
        self._schedule_times: dict[int, float] = get_schedule_times(
            self._deck, time_unit=time_unit
        )

    def _resolve_dimensions(self) -> GridDimensions | None:
        """
        Resolve grid extent from `SPECGRID` (preferred) or `DIMENS`.

        :returns: A `GridDimensions`, or `None`
            if neither keyword is present (decks consisting only of e.g.
            PVT / relperm tables).
        """
        for name in DIMENSION_KEYWORDS:
            record = self._deck.get_first_record_for(name)
            if record is None:
                continue
            keyword = self._registry.get(name)
            if keyword is None:
                continue
            parsed = keyword.parse(self._deck, None)
            if parsed is None:
                continue
            return GridDimensions(nx=int(parsed["nx"]), ny=int(parsed["ny"]), nz=int(parsed["nz"]))
        return None

    def _resolve_unit_system(self) -> UnitSystem | None:
        """
        Resolve the unit system from the RUNSPEC unit keywords.

        Eclipse decks declare exactly one of `FIELD`, `METRIC`, or `LAB`.
        When none is present (e.g. a standalone PVT or relperm include file),
        returns `None`.

        :returns: The `UnitSystem` declared in the deck. `None` if not present.
        """
        for name, unit_system in (
            ("METRIC", UnitSystem.METRIC),
            ("LAB", UnitSystem.LAB),
            ("FIELD", UnitSystem.FIELD),
            ("SI", UnitSystem.SI),
        ):
            if self._deck.has(name):
                return unit_system
        return None

    @property
    def deck(self) -> Deck:
        """The underlying scanned `bores.deck.core.Deck`."""
        return self._deck

    @property
    def keywords(self) -> list[str]:
        """
        Return every unique keyword in the deck, in file order, including
        keywords with no record of their own that are nonetheless resolvable
        because an operator (`EQUALS`/`ADD`/`MULTIPLY`/`COPY`/`MAXVALUE`/
        `MINVALUE`) targets them (e.g. `DZ` set purely via `EQUALS`).

        Operator-only names are appended after the literal ones, sorted,
        since they have no single file position of their own to order by.
        """
        literal = self._deck.keywords
        operator_only = sorted(self._operation_targets - set(literal))
        return literal + operator_only

    def add_keywords(self, *keywords: Keyword[typing.Any]) -> None:
        """
        Register additional keywords (or override existing ones by name).

        Clears any cached value for a keyword name that is being replaced.

        :param keywords: One or more `bores.deck.keywords.base.Keyword`
            instances.
        """
        for keyword in keywords:
            self._registry[keyword.name] = keyword
            self._cache.pop(keyword.name, None)

    def has(self, k: str | Keyword[typing.Any], /) -> bool:
        """
        Return whether `k` occurs anywhere in the deck. Either as an
        explicit record of its own, or as the target of a `BOX`/`EQUALS`/
        `ADD`/`MULTIPLY`/`COPY`/`MAXVALUE`/`MINVALUE` operator record.

        A keyword like `DZ` set purely via `EQUALS 'DZ' ... /` never appears
        as its own record line, but `ArrayKeyword.parse` can still fully
        resolve it from the operator timeline alone, so it counts as present.
        """
        name = k.name if isinstance(k, Keyword) else k.upper()
        return self._deck.has(name) or name in self._operation_targets

    @typing.overload
    def get(self, k: Keyword[T], /, *, use_cache: bool = ...) -> T | None: ...

    @typing.overload
    def get(self, k: str, /, *, use_cache: bool = ...) -> typing.Any | None: ...

    def get(
        self, k: str | Keyword[typing.Any], /, *, use_cache: bool = False
    ) -> typing.Any | None:
        """
        Parse and return the value of keyword `k`.

        :param k: Keyword `k`, case-insensitive or a `Keyword` object (never cached if `Keyword` object).
        :param use_cache: Whether to read/write the per-`DeckFile`
            value cache. Set `False` to force re-parsing (mainly for testing).
        :returns: The keyword's parsed value, or `None` if it is absent
            from the deck or not registered.
        """
        if isinstance(k, Keyword):
            if not self.has(k.name):
                return None
            try:
                return k.parse(
                    self._deck,
                    self.dimensions,
                    operations=self._operations,
                    schedule_times=self._schedule_times,
                )
            except (ValueError, TypeError) as exc:
                raise DeckParseError(f"Could not parse keyword `{k.name}`'s value: {exc}") from exc

        key = k.upper()
        if key not in self._registry or not self.has(key):
            return None

        if use_cache and key in self._cache:
            return self._cache[key]

        try:
            value = self._registry[key].parse(
                self._deck,
                self.dimensions,
                operations=self._operations,
                schedule_times=self._schedule_times,
            )
        except (ValueError, TypeError) as exc:
            raise DeckParseError(f"Could not parse keyword `{key}`: {exc}") from exc

        if use_cache:
            self._cache[key] = value
        return value

    def gets(
        self, *ks: str | Keyword[typing.Any], use_cache: bool = False
    ) -> dict[str, typing.Any | None]:
        """
        Convenience batch form of `get`.

        :param ks: Keyword ks to retrieve.
        :param use_cache: Passed through to `get`.
        :returns: `{name: value}` dict in the order requested.
        """
        return {
            k.name if isinstance(k, Keyword) else k: self.get(k, use_cache=use_cache) for k in ks
        }


DataFile = DeckFile  # Alias
