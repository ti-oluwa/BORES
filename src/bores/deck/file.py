"""Entry point for parsing Eclipse / GRDECL decks."""

import copy
import os
import typing
from collections.abc import Collection
from pathlib import Path

import numpy as np
from typing_extensions import Self, TypeVar

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, report, resolve_source
from bores.deck.keywords.base import ArrayKeyword, Keyword, get_schedule_clock
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
    INIT,
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
    AQUTAB,
    DENSITY,
    EHYSTR,
    GRAVITY,
    PVCDO,
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
    COMPSEGS,
    DATES,
    DRSDT,
    DRVDT,
    GCONINJE,
    GCONPROD,
    GECON,
    GRUPTREE,
    TSTEP,
    VFPINJ,
    VFPPROD,
    WCONHIST,
    WCONINJE,
    WCONINJH,
    WCONPROD,
    WDFAC,
    WECON,
    WELOPEN,
    WELPI,
    WELSEGS,
    WELSPECS,
    WELTARG,
    WGRUPCON,
    WLIFT,
    WPAVE,
    WPIMULT,
    WTEST,
)
from bores.deck.keywords.solution import (
    AQUANCON,
    AQUCT,
    AQUFETP,
    AQUFLUX,
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
    BGKR,
    BGSAT,
    BOKR,
    BOSAT,
    BPR,
    BPRESSUR,
    BRS,
    BRV,
    BSGAS,
    BSOIL,
    BSWAT,
    BWKR,
    BWSAT,
    FGIR,
    FGIT,
    FGOR,
    FGPR,
    FGPT,
    FLPR,
    FOIR,
    FOIT,
    FOPR,
    FOPT,
    FWCT,
    FWIR,
    FWIT,
    FWPR,
    FWPT,
    RGIP,
    ROIP,
    RPTRST,
    RPTSCHED,
    RWIP,
    WBHP,
    WGIR,
    WGIT,
    WGOR,
    WGPR,
    WGPT,
    WLPR,
    WOIR,
    WOIT,
    WOPR,
    WOPT,
    WTHP,
    WWCT,
    WWIR,
    WWIT,
    WWPR,
    WWPT,
)
from bores.deck.operators import Operation, resolve_operations
from bores.types import TextOrPath, UnitSystem

__all__ = ["DEFAULT_KEYWORDS", "DataFile", "DeckFile"]

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
    INIT,
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
    EHYSTR,
    GRAVITY,
    PVCDO,
    PVCO,
    PVDG,
    PVDO,
    PVTG,
    PVTO,
    AQUTAB,
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
    AQUANCON,
    AQUCT,
    AQUFETP,
    AQUFLUX,
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
    COMPSEGS,
    DATES,
    DRSDT,
    DRVDT,
    GCONINJE,
    GCONPROD,
    GRUPTREE,
    TSTEP,
    WECON,
    WELOPEN,
    WELSEGS,
    WELSPECS,
    WELTARG,
    WLIFT,
    VFPINJ,
    VFPPROD,
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
    BGKR,
    BGSAT,
    BOKR,
    BOSAT,
    BPR,
    BPRESSUR,
    BRS,
    BRV,
    BSGAS,
    BSOIL,
    BSWAT,
    BWKR,
    BWSAT,
    FGIR,
    FGIT,
    FGOR,
    FGPR,
    FGPT,
    FLPR,
    FOIR,
    FOIT,
    FOPR,
    FOPT,
    FWCT,
    FWIR,
    FWIT,
    FWPR,
    FWPT,
    RGIP,
    ROIP,
    RPTRST,
    RPTSCHED,
    RWIP,
    WBHP,
    WGIR,
    WGIT,
    WGOR,
    WGPR,
    WGPT,
    WLPR,
    WOIR,
    WOIT,
    WOPR,
    WOPT,
    WTHP,
    WWCT,
    WWIR,
    WWIT,
    WWPR,
    WWPT,
})


class DeckFile:
    """
    A parsed Eclipse deck (*.DATA / *.GRDECL) file.

    Comment-stripped, `INCLUDE`-resolved text plus a registry of `bores.deck.keywords.base.Keyword`
    objects, exposed through `get`.

    **Usage**:

    ```python
    from bores.deck import DeckFile, DEFAULT_KEYWORDS
    from bores.deck import keywords as k

    df = DeckFile("path/to/model.DATA")
    poro = df.get(
        "PORO"
    )  # typing.Any but actually np.ndarray (n_cells,) or None (poor typing support)
    faults = df.get(k.FAULTS)  # list[dict[str, typing.Any]] or None (better typing support)

    # With extra keywords not in the default set:
    df2 = DeckFile(text, keywords=DEFAULT_KEYWORDS | {MyCustomKeyword()})
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
        "_report_times",
        "_schedule_times",
        "dimensions",
        "strict",
        "unit_system",
    )

    def __init__(
        self,
        source: TextOrPath,
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        encoding: str = "ascii",
        unit_system: UnitSystem | None = None,
        strict: bool = False,
    ) -> None:
        """
        :param source: A file path, raw deck text `str`, or raw deck
            `bytes`. Filesystem paths and `Path` objects are read from
            disk; `INCLUDE` directives are resolved relative to the source
            file's directory. Raw text input drops `INCLUDE` directives with a warning.
            A `str` of more than one line is raw deck text. A single-line `str` is read as a file
            if one exists at that path, and otherwise as raw deck text if it is a lone keyword
            (such as `METRIC`); anything else is a missing-file error. To be explicit, use
            `from_text`, `from_file` or `from_bytes`.
        :param keywords: Supported `bores.deck.keywords.base.Keyword`
            instances. Only supported keywords can be read from the file.
        :param encoding: Text encoding for file/bytes input (default `"ascii"`).
        :param unit_system: The unit system to use for the deck. If not provided,
            the deck is scanned for a `FIELD`, `METRIC`, `LAB` or `SI` keyword.
            If none is found, a `DeckParseError` is raised.
            Do not provide this argument if you want the unit system to be auto-detected.
        :param strict: How to treat input that is malformed or ambiguous but that a lenient parse
            can still make something of. When `False` (the default) a warning is emitted and the
            parser falls back to a documented behaviour. When `True` a `DeckParseError` is raised
            instead. This covers: data without a terminating `/`, an unclosed quote, a keyword
            that may appear once appearing again, `SPECGRID` and `DIMENS` disagreeing, several
            unit-system keywords, an `INCLUDE` in raw text, an unreadable `BOX` or operator
            record, and (when the arrays are read) a single value standing in for a whole array,
            defaulted (`N*`) array entries and a malformed `START`, `DATES` or `TSTEP`.
            Asking for a keyword that is not registered, or that is absent, is never an error:
            `get` returns `None`.
        :raises DeckParseError: If a file cannot be read, or the deck is malformed (see `strict`).
        """
        self.strict = strict
        self._deck = Deck(resolve_source(source, encoding=encoding, strict=strict), strict=strict)

        self._registry = {keyword.name: keyword for keyword in keywords}
        self._cache: dict[str, typing.Any] = {}
        self.dimensions = self._resolve_dimensions()
        deck_unit_system = unit_system if unit_system is not None else self._resolve_unit_system()
        if deck_unit_system is None:
            raise DeckParseError(
                "Could not detect deck unit system. Perhaps you passed a standalone PVT, GRDECL or relperm include file. "
                "Provide `unit_system` on instantiation. If `source` was meant as a file path, "
                "check that the file exists."
            )

        self.unit_system = deck_unit_system
        self._operations: list[Operation] | None = None
        self._operation_targets: frozenset[str] = frozenset()
        self._resolve_operations()

        time_unit = "days"
        if deck_unit_system == UnitSystem.LAB:
            time_unit = "hours"
        elif deck_unit_system == UnitSystem.SI:
            time_unit = "seconds"
        self._schedule_times, self._report_times = get_schedule_clock(
            self._deck, time_unit=time_unit
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        unit_system: UnitSystem | None = None,
        strict: bool = False,
    ) -> Self:
        """
        Parse raw deck text. Unlike the plain constructor this never treats `text` as a path, even
        when it is a single line (such as `"METRIC"`).

        `INCLUDE` directives cannot be followed from raw text; see `strict`.

        :param text: The deck text.
        :param keywords: See `DeckFile`.
        :param unit_system: See `DeckFile`.
        :param strict: See `DeckFile`.
        """
        if not isinstance(text, str):
            raise TypeError(f"`text` must be a str, got {type(text).__name__}.")

        # A trailing newline makes a one-line text unambiguously raw text, and changes nothing else.
        return cls(
            text if ("\n" in text or "\r" in text) else text + "\n",
            keywords=keywords,
            unit_system=unit_system,
            strict=strict,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        encoding: str = "ascii",
        unit_system: UnitSystem | None = None,
        strict: bool = False,
    ) -> Self:
        """
        Parse a deck file. `INCLUDE` directives are resolved relative to the file's directory.

        :param path: Path of the file.
        :param keywords: See `DeckFile`.
        :param encoding: Text encoding of the file and of every included file.
        :param unit_system: See `DeckFile`.
        :param strict: See `DeckFile`.
        :raises DeckParseError: If the file does not exist or cannot be read.
        """
        return cls(
            Path(path),
            keywords=keywords,
            encoding=encoding,
            unit_system=unit_system,
            strict=strict,
        )

    from_path = from_file

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        keywords: Collection[Keyword[typing.Any]] = DEFAULT_KEYWORDS,
        encoding: str = "ascii",
        unit_system: UnitSystem | None = None,
        strict: bool = False,
    ) -> Self:
        """
        Parse raw deck bytes. `INCLUDE` directives cannot be followed from bytes; see `strict`.

        :param data: The encoded deck text.
        :param keywords: See `DeckFile`.
        :param encoding: Text encoding of `data`.
        :param unit_system: See `DeckFile`.
        :param strict: See `DeckFile`.
        """
        return cls(
            data,
            keywords=keywords,
            encoding=encoding,
            unit_system=unit_system,
            strict=strict,
        )

    @property
    def report_times(self) -> list[float]:
        """
        Every report time the deck asks for, in the deck's own time unit (days for `FIELD` and
        `METRIC`, hours for `LAB`, seconds for `SI`): `0.0`, each entry of every `TSTEP` and each
        date of every `DATES`, sorted and without repeats. The last one is where the deck's
        schedule ends.
        """
        return list(self._report_times)

    def _resolve_dimensions(self) -> GridDimensions | None:
        """
        Resolve grid extent from `SPECGRID` (preferred) or `DIMENS`.

        If both are present and disagree, `SPECGRID` is used and the conflict is reported
        (an error when `strict`).

        :returns: A `GridDimensions`, or `None`
            if neither keyword is present (decks consisting only of e.g.
            PVT / relperm tables).
        """
        declared: list[tuple[str, GridDimensions]] = []
        for name in DIMENSION_KEYWORDS:
            if self._deck.get_first_record_for(name) is None:
                continue
            keyword = self._registry.get(name)
            if keyword is None:
                continue
            parsed = keyword.parse(self._deck, None)
            if parsed is None:
                continue
            declared.append((
                name,
                GridDimensions(nx=int(parsed["nx"]), ny=int(parsed["ny"]), nz=int(parsed["nz"])),
            ))

        if not declared:
            return None

        chosen_name, chosen = declared[0]
        for name, other in declared[1:]:
            if other != chosen:
                report(
                    f"`{chosen_name}` declares a {chosen.nx}x{chosen.ny}x{chosen.nz} grid but "
                    f"`{name}` declares {other.nx}x{other.ny}x{other.nz}. Using `{chosen_name}`.",
                    strict=self.strict,
                    stacklevel=3,
                )
        return chosen

    def _resolve_unit_system(self) -> UnitSystem | None:
        """
        Resolve the unit system from the `RUNSPEC` unit keywords.

        Eclipse decks declare exactly one of `FIELD`, `METRIC`, or `LAB` (`SI` is also recognised
        as a BORES extension). When none is present (e.g. a standalone PVT or relperm include
        file), returns `None`. When several are present the deck is contradicting itself: that is
        reported (an error when `strict`) and, if parsing carries on, the first of `METRIC`,
        `LAB`, `FIELD`, `SI` wins.

        :returns: The `UnitSystem` declared in the deck. `None` if not present.
        """
        declared = [
            (name, unit_system)
            for name, unit_system in (
                ("METRIC", UnitSystem.METRIC),
                ("LAB", UnitSystem.LAB),
                ("FIELD", UnitSystem.FIELD),
                ("SI", UnitSystem.SI),
            )
            if self._deck.has(name)
        ]
        if not declared:
            return None
        if len(declared) > 1:
            names = ", ".join(f"`{name}`" for name, _ in declared)
            report(
                f"The deck declares more than one unit system ({names}). "
                f"Using `{declared[0][0]}`.",
                strict=self.strict,
                stacklevel=3,
            )
        return declared[0][1]

    def _array_info(self) -> dict[str, tuple[float, tuple[str, ...] | None]]:
        """What a `COPY` needs to know about each registered array keyword to read it as a source."""
        return {
            name: (
                keyword.default_value,
                tuple(keyword.column_shape) if keyword.column_shape else None,
            )
            for name, keyword in self._registry.items()
            if isinstance(keyword, ArrayKeyword)
        }

    def _resolve_operations(self) -> None:
        """(Re)resolve the `BOX` / operator records, which depend on the grid dimensions."""
        self._operations = (
            resolve_operations(self._deck, self.dimensions, array_info=self._array_info())
            if self.dimensions is not None
            else None
        )
        self._operation_targets = (
            frozenset(operation.target for operation in self._operations)
            if self._operations
            else frozenset()
        )

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

        Clears any cached value for a keyword name that is being replaced. Registering a grid
        dimension keyword (`SPECGRID` / `DIMENS`), or an array keyword that a `COPY` reads from,
        re-resolves the operators that depend on it, and so clears the whole cache.

        :param keywords: One or more `bores.deck.keywords.base.Keyword`
            instances.
        """
        for keyword in keywords:
            self._registry[keyword.name] = keyword
            self._cache.pop(keyword.name, None)

        dimensions_changed = any(keyword.name in DIMENSION_KEYWORDS for keyword in keywords)
        if dimensions_changed:
            self.dimensions = self._resolve_dimensions()

        # An operator's `COPY` records how to read its source array, so a source registered (or
        # replaced) now needs the operators resolved again.
        copy_sources = {op.source for op in self._operations or () if op.source is not None}
        if dimensions_changed or any(
            isinstance(keyword, ArrayKeyword) and keyword.name in copy_sources
            for keyword in keywords
        ):
            self._resolve_operations()
            self._cache.clear()

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
            value cache. Off by default, so every call parses afresh and returns an array of its
            own. Set `False` to force re-parsing (mainly for testing).

            Warning:
                A cached value is shared by every caller, so changing it in place would change
                what every later `get` returns. To make that impossible for arrays, a cached
                `numpy` array is read-only (writing to it raises `ValueError`; take a `.copy()`
                to modify it). Any other cached value (a list or dict) is copied on every read,
                so changing it only changes your copy. With `use_cache=False` each call returns a
                fresh, writable array.

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
            cached = self._cache[key]
            return cached if isinstance(cached, np.ndarray) else copy.deepcopy(cached)

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
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
                self._cache[key] = value
            else:
                self._cache[key] = copy.deepcopy(value)
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
