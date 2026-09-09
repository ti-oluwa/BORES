"""
Base implementation for Eclipse keyword parsers, and concrete implementations
for various common keyword patterns.
"""

import abc
import datetime
import typing
import warnings
from collections.abc import Collection

import attrs
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.datastructures import GridDimensions
from bores.deck.core import (
    Deck,
    DeckParseError,
    parse_repeat_token,
    tokenize,
)
from bores.deck.operators import Operation, apply_operation, resolve_operations
from bores.types import FloatArray, Number, OneDimension

__all__ = [
    "ArrayKeyword",
    "DateKeyword",
    "DatesKeyword",
    "Field",
    "FlagKeyword",
    "Keyword",
    "RecordKeyword",
    "RepeatedRecordKeyword",
    "TableKeyword",
]

T = typing.TypeVar("T")


@attrs.frozen(slots=True, frozen=True)
class Field(typing.Generic[T]):
    """
    One positional field in a `RecordKeyword` or `RepeatedRecordKeyword` layout.

    :param name: Field name (used as key in the returned `dict`).
    :param type: Callable that converts the raw token string to the field
        value (e.g. `int`, `float`, `str`, or a function/validator).
    :param required: Whether this field must be present.  Trailing optional
        fields fall back to `default` when the record has fewer tokens.
    :param default: Default value when the field is absent and
        `required=False`.
    """

    name: str
    type: typing.Callable[[str], T]
    required: bool = True
    default: T | None = None
    options: Collection[T] | None = None

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValueError("Field name required. `name` can not be empty.")

        options = frozenset(self.options) if self.options else None
        if options and self.default is not None and self.default not in options:
            raise ValueError(
                f"Invalid default value for {self.name!r} - {self.default!r}. "
                f"Default should be one of {', '.join([str(option) for option in options])}"
            )
        object.__setattr__(self, "options", options)

    def parse(self, raw: str | None, keyword: str) -> T | None:
        """
        Parse one Eclipse field value.

        - `raw=None` means the field was omitted entirely.
        - `raw="1*"` is Eclipse's explicit "use the default value" designator.

        :param raw: Raw token string, or `None` if the field is absent.
        :param keyword: Keyword name for error messages.
        :returns: Parsed value (or the field default).
        :raises DeckParseError: If the field is missing, contains an invalid
            value, or violates the declared option set.
        """
        # Field omitted entirely.
        if raw is None:
            if self.required:
                raise DeckParseError(f"{keyword} record: missing required field {self.name!r}.")
            return self.default

        # Eclipse explicit default ("1*").
        if raw == "1*":
            if self.required:
                raise DeckParseError(
                    f"{keyword} record: required field {self.name!r} "
                    "cannot use the default designator ('1*')."
                )
            return self.default

        try:
            value = self.type(raw)
        except (TypeError, ValueError) as exc:
            raise DeckParseError(
                f"{keyword} record: {self.name!r} got invalid value {raw!r}: {exc}"
            ) from exc

        if self.options is not None and value not in self.options:
            raise DeckParseError(
                f"{keyword} record: {self.name!r} got unrecognised value {raw!r}. "
                f"Value should be one of "
                f"{', '.join(map(str, self.options))}."
            )
        return value


class Keyword(typing.Generic[T], abc.ABC):
    """
    Base class for a single supported Eclipse keyword.

    A `Keyword` is pure parsing / shape logic: given the
    `bores.deck.core.Record`(s) that belong to it (already located by
    `bores.deck.core.Deck`) plus the deck's resolved grid dimensions,
    produce the keyword's Python value.

    Each concrete subclass only has to declare *shape*: field names/types
    for record keywords, expected array length for grid-array keywords, or
    nothing extra for nullary keywords. All cross-cutting deck mechanics
    (comments, includes, repeats, operators, boxing, caching) live in
    `bores.deck.file.DeckFile` and are shared by every
    keyword for free.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name.upper()

    @abc.abstractmethod
    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> T | None:
        """
        Parse this keyword's value out of `deck`.

        :param deck: The scanned deck to read records from.
        :param dims: Resolved grid dimensions, or `None` if
            `SPECGRID` / `DIMENS` has not been parsed yet (some
            keywords, e.g. `MAPAXES`, do not need it; array keywords
            will raise if it is required and missing).
        :returns: The parsed value, or `None` if the keyword is absent
            from the deck.
        """
        ...

    def __hash__(self) -> int:
        return hash(self.name)


def parse_tokens(
    keyword: str, fields: typing.Sequence[Field[T]], tokens: typing.Sequence[str]
) -> dict[str, T | None]:
    """
    Convert a flat token sequence to a `{field_name: value}` dict
    according to `fields`.

    :param keyword: Keyword name, used only in error messages.
    :param fields: Field descriptors in positional order.
    :param tokens: Already-expanded token list for this record.
    :returns: Parsed field dict.
    :raises DeckParseError: If a required field is missing or has an
        invalid value.
    """
    result: dict[str, T | None] = {}
    n_tokens = len(tokens)
    for idx, field in enumerate(fields):
        raw = tokens[idx] if idx < n_tokens else None
        result[field.name] = field.parse(raw, keyword)
    return result


class RecordKeyword(Keyword[dict[str, T | None]]):
    """
    A keyword holding exactly one fixed-layout record of mixed-type fields
    (`SPECGRID`, `MAPAXES`, `GRIDUNIT`, `PINCH`).

    Subclasses declare `fields: Sequence[Field]` describing each
    positional token; `parse` returns a `dict` keyed by field name.
    Trailing optional fields (`required=False`) fall back to their
    `default` when the deck record has fewer tokens than declared fields.
    """

    __slots__ = ("fields",)

    def __init__(self, name: str, fields: typing.Sequence[Field[T]]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> dict[str, T | None] | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        # `record.body` spans up to the next keyword line, since some
        # keywords need multiple internal `/` segments. We only
        # want the first (and only) `/`-terminated record here.
        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        return self.parse_tokens(tokens)

    def parse_tokens(self, tokens: typing.Sequence[str]) -> dict[str, T | None]:
        return parse_tokens(self.name, self.fields, tokens)


class RepeatedRecordKeyword(Keyword[list[dict[str, T | None]]]):
    """
    A keyword whose body holds zero or more individually `/`-terminated
    records sharing one field layout (`FAULTS`, `MULTFLT`, `NNC`).

    Multiple occurrences of the keyword in the deck (e.g. several
    `FAULTS` blocks) are concatenated in file order.

    Note:
        The body is split on `"/"` before tokenising individual lines.
        Quoted field values that contain a literal `"/"` character are
        therefore not supported (they are vanishingly rare in Eclipse
        practice).
    """

    __slots__ = ("fields",)

    def __init__(self, name: str, fields: typing.Sequence[Field[T]]) -> None:
        super().__init__(name)
        self.fields = list(fields)

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[dict[str, T | None]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        results: list[dict[str, T | None]] = []
        for record in records:
            for line in record.body.split("/"):
                tokens = tokenize(line)
                if not tokens:
                    continue
                results.append(self.parse_tokens(tokens))
        return results or None

    def parse_tokens(self, tokens: typing.Sequence[str]) -> dict[str, T | None]:
        return parse_tokens(self.name, self.fields, tokens)


class ArrayKeyword(Keyword[FloatArray[OneDimension]]):
    """
    A flat per-cell array keyword of length `nx * ny * nz`
    (`PORO`, `PERMX`, `ACTNUM`, `MULTX`, `TOPS`, …).

    Handles `N*value` repeat expansion and is automatically a valid
    target for `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY` /
    `MAXVALUE` / `MINVALUE` operator records anywhere in the deck.
    Operator resolution is shared across all instances via
    `bores.deck.file.DeckFile`; no individual keyword
    subclass needs to implement it.

    If the keyword itself never appears as a standalone data block but *is*
    the target of one or more operator records (a common Eclipse idiom for
    keywords like `MULTX` that are only ever set via `BOX`/`EQUALS`),
    the array still starts from `default_value` and operators are applied
    on top.
    """

    __slots__ = ("column_shape", "default_value", "dtype", "is_multiplier")

    def __init__(
        self,
        name: str,
        *,
        dtype: npt.DTypeLike = np.float64,
        default_value: float = 0.0,
        is_multiplier: bool = False,
        column_shape: tuple[str, ...] | None = None,
    ) -> None:
        """
        :param name: Keyword name (e.g. `"PORO"`).
        :param dtype: Output array dtype. The internal accumulation is
            always `float64`; the result is cast to `dtype` at the end.
        :param default_value: Value cells start at before any explicit data
            block or operator is applied.
        :param is_multiplier: Convenience flag; when `True`, overrides
            `default_value` to `1.0` (the multiplicative identity)
            regardless of what was passed for `default_value`.  Use for
            `MULTX`, `MULTY`, `MULTZ`, and their `-` variants.
        :param column_shape: Optional tuple of dimension-name strings declaring
            a valid "short form" token count for this keyword.  Each string
            must be one of `"nx"`, `"ny"`, `"nz"`.  When the token count
            equals the product of the named dimensions, the parsed values are
            broadcast (tiled) along the remaining axes to fill `n_cells`.

            **For example** - `TOPS` accepts either `nx*ny` column tops or a full
            `nx*ny*nz` per-cell array:

            ```python
            TOPS = ArrayKeyword("TOPS", column_shape=("nx", "ny"))
            ```

            Supported broadcast patterns (axes not listed in `column_shape`
            are tiled):

            ```
            ======================  ========================  =====================
            `column_shape`          Short-form token count    Broadcast direction
            ======================  ========================  =====================
            `("nx", "ny")`          `nx * ny`                 repeated `nz` times
            `("nx", "nz")`          `nx * nz`                 repeated `ny` times
            `("ny", "nz")`          `ny * nz`                 repeated `nx` times
            `("nx",)`               `nx`                      repeated `ny*nz` times
            `("ny",)`               `ny`                      repeated `nx*nz` times
            `("nz",)`               `nz`                      repeated `nx*ny` times
            ======================  ========================  =====================
            ```
        """
        super().__init__(name)
        self.dtype = np.dtype(dtype)
        self.default_value = 1.0 if is_multiplier else default_value
        self.is_multiplier = is_multiplier

        if column_shape and len(column_shape) > 3:
            raise ValueError(f"Invalid size for `column_shape`: {column_shape!r}")
        self.column_shape = [column.lower() for column in column_shape] if column_shape else None

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> FloatArray[OneDimension] | None:
        """
        Parse and return the resolved per-cell array.

        :param deck: The scanned deck.
        :param dims: Resolved grid dimensions (required).
        :returns: Shape `(n_cells,)` array in Eclipse flat order
            (i-fastest), or `None` if the keyword is absent from the
            deck and no operator targets it.
        :raises DeckParseError: If `dims` is `None` or the keyword
            body is malformed.
        """
        if dims is None:
            raise DeckParseError(
                f"Cannot parse grid array keyword {self.name!r} without "
                "resolved grid dimensions (`SPECGRID`/`DIMENS` not found)."
            )
        return self._resolve(deck, dims, operations=operations, stop_before_order=None)

    def _resolve(
        self,
        deck: Deck,
        dims: GridDimensions,
        *,
        operations: list[Operation] | None = None,
        stop_before_order: tuple[int, int] | None,
    ) -> FloatArray[OneDimension] | None:
        """
        Replay every event affecting this keyword's array - its own
        explicit data block(s) *and* every operator record targeting it -
        strictly in the order they occur in the deck.

        This is what makes e.g. `MULTX` set via `EQUALS` and later
        overwritten by an explicit `MULTX ... /` block (or vice versa)
        behave like Eclipse, that is, whichever event is last in the file wins for
        the cells it touches.

        :param deck: Scanned deck.
        :param dims: Resolved grid dimensions.
        :param stop_before_order: If given, only replay events with
            `order < stop_before_order`. Used by `COPY` to resolve a
            source keyword's value *as of* the point in the file where the
            `COPY` record itself occurs.
        :returns: The resolved array, or `None` if this keyword has no
            data block and no operator ever targets it.
        """
        events = self._timeline(deck, dims, operations=operations)
        if stop_before_order is not None:
            events = [event for event in events if event[0] < stop_before_order]
        if not events:
            return None

        array = np.full(dims.n_cells, self.default_value, dtype=np.float64)

        def _resolve_source_at(
            name: str,
            as_of_order: tuple[int, int],
            *,
            operations: list[Operation] | None = None,
        ) -> FloatArray[OneDimension] | None:
            # Guess the default_value for arbitrary source keywords:
            # multiplier-style arrays default to 1.0, property arrays to 0.0.
            # We cannot know for certain without a registered Keyword instance,
            # but this heuristic is correct for all standard Eclipse arrays.
            default = 1.0 if name.startswith("MULT") else 0.0
            probe = ArrayKeyword(name, default_value=default)
            return probe._resolve(deck, dims, operations=operations, stop_before_order=as_of_order)

        for order, kind, payload in events:
            if kind == "assign":
                tokens: list[str] = payload  # type: ignore[assignment]
                if len(tokens) == 0:
                    warnings.warn(
                        f"{self.name}: assign event has zero tokens; skipping.",
                        stacklevel=4,
                    )
                    continue

                # Total element count this assign event actually represents,
                # accounting for unexpanded "N*value" groups, without building the expanded list.
                total_count = sum((parse_repeat_token(token) or (1, None))[0] for token in tokens)
                if total_count == 1:
                    try:
                        array[:] = float(tokens[0])
                    except ValueError as exc:
                        raise DeckParseError(
                            f"{self.name}: expected a numeric scalar, got {tokens[0]!r}: {exc}"
                        ) from exc

                elif total_count == dims.n_cells:
                    try:
                        position = 0
                        for token in tokens:
                            repeat = parse_repeat_token(token)
                            if repeat is not None:
                                count, value = repeat
                                array[position : position + count] = float(value)
                                position += count
                            else:
                                array[position] = float(token)
                                position += 1
                    except ValueError as exc:
                        raise DeckParseError(
                            f"{self.name} contains a non-numeric value: {exc}"
                        ) from exc

                else:
                    # Try `column_shape` broadcast if declared. Short-form
                    # arrays are small by construction (nx*ny at most), so
                    # expanding here is cheap so we reuse the existing path via
                    # a fully-expanded token list for this branch only.
                    broadcast_ok = False
                    if self.column_shape is not None:
                        short_count = self._short_form_count(dims)
                        if total_count == short_count:
                            expanded: list[str] = []
                            for token in tokens:
                                repeat = parse_repeat_token(token)
                                if repeat is not None:
                                    expanded.extend([repeat[1]] * repeat[0])
                                else:
                                    expanded.append(token)
                            try:
                                array[:] = self._broadcast_short_form(
                                    np.array(expanded, dtype=np.float64), dims
                                )
                                broadcast_ok = True
                            except ValueError as exc:
                                raise DeckParseError(
                                    f"{self.name} contains a non-numeric value: {exc}"
                                ) from exc

                    if not broadcast_ok:
                        expected_desc = f"1 or {dims.n_cells}"
                        if self.column_shape is not None:
                            short_count = self._short_form_count(dims)
                            expected_desc += (
                                f" or {short_count} (short form: {' * '.join(self.column_shape)})"
                            )
                        raise DeckParseError(
                            f"{self.name} expected {expected_desc} value(s); got {total_count}."
                        )

            else:  # kind == "operate"
                operation: Operation = payload  # type: ignore[assignment]
                apply_operation(
                    array,
                    operation,
                    dims,
                    resolve_source=lambda name, _order=order: _resolve_source_at(
                        name, _order, operations=operations
                    ),
                )
        return array.astype(self.dtype, copy=False)  # type: ignore[return-value]

    def _short_form_count(self, dims: GridDimensions) -> int:
        """
        Return the number of tokens accepted in the short (column) form.

        :param dims: Resolved grid dimensions.
        :returns: Product of the dimensions named in `self.column_shape`.
        """
        assert self.column_shape is not None
        dim_map = {"nx": dims.nx, "ny": dims.ny, "nz": dims.nz}
        result = 1
        for axis in self.column_shape:
            result *= dim_map[axis]
        return result

    def _broadcast_short_form(
        self,
        short_array: npt.NDArray[np.float64],
        dims: GridDimensions,
    ) -> npt.NDArray[np.float64]:
        """
        Broadcast a short-form array to full `n_cells` length.

        The short array covers the axes in `self.column_shape` in Eclipse
        flat order (i fastest). The missing axes are tiled so that the
        result is in standard Eclipse flat order `i + j*nx + k*nx*ny`.

        :param short_array: 1-D float64 array of length `_short_form_count(dims)`.
        :param dims: Resolved grid dimensions.
        :returns: 1-D float64 array of length `n_cells`.
        """
        assert self.column_shape is not None
        dim_map = {"nx": dims.nx, "ny": dims.ny, "nz": dims.nz}

        # Reshape short array into its natural (i,j,k) sub-space
        shape_present = tuple(dim_map[ax] for ax in self.column_shape)
        array = short_array.reshape(shape_present, order="F")  # i fastest

        # Expand missing axes by inserting new dimensions and tiling
        # We build the full (nx, ny, nz) array then ravel in Fortran order
        full = np.empty((dims.nx, dims.ny, dims.nz), dtype=np.float64)

        # Map axis names to indices in the full (nx,ny,nz) array
        axis_indices = {"nx": 0, "ny": 1, "nz": 2}

        # Reshape array to align with full array dimensions
        # Insert size-1 dims for missing axes, then broadcast
        target_shape = [1, 1, 1]
        for _local_i, ax in enumerate(self.column_shape):
            target_shape[axis_indices[ax]] = int(dim_map[ax])
        arr_expanded = array.reshape(target_shape)
        full[:] = np.broadcast_to(arr_expanded, (int(dims.nx), int(dims.ny), int(dims.nz)))
        return full.ravel(order="F")  # Eclipse flat order: i fastest

    def _timeline(
        self,
        deck: Deck,
        dims: GridDimensions,
        *,
        operations: list[Operation] | None = None,
    ) -> list[tuple[tuple[int, int], str, typing.Any]]:
        """
        Build the ordered `(order, kind, payload)` events affecting this
        keyword: `("assign", tokens)` for each of its own explicit
        data-block records, interleaved with `("operate", Operation)`
        for every operator record that targets it.

        Both event kinds share one `order` namespace - a
        `(record_start, line_index)` tuple.

        :returns: Events sorted ascending (earliest in file first).
        """
        events: list[tuple[tuple[int, int], str, typing.Any]] = []

        for record in deck.get_records_for(self.name):
            # Compact tokens: an "N*value" repeat group stays as one token
            # instead of expanding to N copies - large arrays are commonly
            # one or a handful of such groups, so this avoids materializing
            # (and then redundantly re-parsing) millions of identical
            # strings for e.g. a uniform full-field PORO/PERMX.
            raw_tokens = tokenize(record.body.split("/", 1)[0], expand_repeats=False)
            # A bare "N*" (empty value) repeat group is dropped with a
            # warning rather than crashing float() later.
            clean_tokens = []
            dropped = 0
            for token in raw_tokens:
                repeat = parse_repeat_token(token)
                if repeat is not None and repeat[1] == "":
                    dropped += repeat[0]
                    continue
                clean_tokens.append(token)

            if dropped:
                warnings.warn(
                    f"{self.name}: record body contains bare 'N*' repeat(s) "
                    f"(empty-value tokens); {dropped} token(s) dropped.",
                    stacklevel=6,
                )
            raw_tokens = clean_tokens
            events.append(((record.start, 0), "assign", raw_tokens))

        if operations is None:
            operations = resolve_operations(deck, dims)

        for operation in operations:
            if operation.target == self.name:
                events.append((operation.order, "operate", operation))

        events.sort(key=lambda e: e[0])
        return events


MONTH_MAP: dict[str, int] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _parse_date(tokens: typing.Sequence[str], name: str) -> datetime.date:
    """
    Parse a three-token Eclipse date `[day, month, year]` into a `datetime.date`.

    Eclipse month tokens may be bare (`JAN`) or quoted (`'JAN'`);
    quoting is already stripped by `bores.deck.core.tokenize`.

    :param tokens: At least three string tokens: day, month abbreviation, year.
    :param name: Keyword name for error messages.
    :returns: Parsed date.
    :raises DeckParseError: If the tokens cannot be parsed as a valid date.
    """
    if len(tokens) < 3:
        raise DeckParseError(
            f"{name}: expected 3 tokens for a date (DAY MON YEAR); "
            f"got {len(tokens)}: {list(tokens)!r}."
        )
    try:
        day = int(tokens[0])
    except ValueError as exc:
        raise DeckParseError(f"{name}: day token {tokens[0]!r} is not an integer.") from exc

    month_str = tokens[1].upper()
    month = MONTH_MAP.get(month_str)
    if month is None:
        raise DeckParseError(
            f"{name}: unrecognised month abbreviation {tokens[1]!r}. "
            f"Expected one of {sorted(MONTH_MAP)}."
        )

    try:
        year = int(tokens[2])
    except ValueError as exc:
        raise DeckParseError(f"{name}: year token {tokens[2]!r} is not an integer.") from exc

    try:
        return datetime.date(year, month, day)
    except ValueError as exc:
        raise DeckParseError(f"{name}: invalid date {day}/{month}/{year}: {exc}") from exc


class FlagKeyword(Keyword[bool]):
    """
    A bare presence-only switch keyword that carries no data section.

    Examples: `OIL`, `GAS`, `WATER`, `DISGAS`, `VAPOIL`,
    `FIELD`, `METRIC`, `LAB`, `NOSIM`, `UNIFIN`, `UNIFOUT`.

    `parse` returns `True` when the keyword appears anywhere in
    the deck, `False` otherwise.  It never returns `None` so callers
    can always do a truthiness test.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> bool:
        """
        :returns: `True` if the keyword is present in the deck,
            `False` if absent.
        """
        return deck.has(self.name)


class DateKeyword(Keyword[datetime.date]):
    """
    A keyword whose single record is an Eclipse-format date
    `D MON YYYY /`.

    Used for `START`.

    `parse` returns `datetime.date`, or `None` when the
    keyword is absent.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> datetime.date | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        return _parse_date(tokens, self.name)


class DatesKeyword(Keyword[list[datetime.date]]):
    """
    The `DATES` keyword: a sequence of one or more `/`-terminated
    date entries inside one `DATES … /` block, each of the form
    `D MON YYYY /`.

    Multiple `DATES` blocks in the same deck are concatenated in file
    order and returned as a single flat list.

    `parse` returns a list of `datetime.date` objects, or
    `None` when the keyword is absent.

    Example deck fragment:

        DATES
         1 JAN 2020 /
         1 FEB 2020 /
        /
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[datetime.date] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        dates: list[datetime.date] = []
        for record in records:
            for segment in record.body.split("/"):
                tokens = tokenize(segment)
                if not tokens:
                    continue
                dates.append(_parse_date(tokens, self.name))
        return dates or None


PVTRow = dict[str, T]
"""One row of a PVT/saturation table: a `{column_name: value}` dict."""

PVTTable = list[PVTRow[T]]
"""One saturation/PVT table: a list of row dicts in ascending primary-key order."""


class TableKeyword(Keyword[list[PVTTable[Number]]]):
    """
    A keyword whose body contains one or more tabulated data blocks,
    each terminated by `/`. Multiple keyword occurrences (e.g. one per
    PVT region) are collected in order.

    This covers two sub-patterns:

    **Simple (immiscible) tables**: every `/`-terminated segment is one
    complete row; all rows in a block share the same column layout.
    Examples: `SWOF`, `SGOF`, `PVDG`, `PVDO`, `SWFN`, `SGFN`,
    `SOF2`, `SOF3`, `ROCK`, `DENSITY`, `PVTW`.

    **Miscible (saturated/under-saturated) tables**: a single primary-key
    value on its own `/`-separated segment introduces a new "inner table",
    and subsequent segments are rows belonging to that inner table until the
    next primary-key segment or a double `//` terminates the table.
    Examples: `PVTO`, `PVTG`.

    :param name: Keyword name.
    :param columns: Column descriptors for each field in a data row.
    :param primary_key: If not `None`, this column name is the
        "outer" key used in miscible (bracketed) PVT tables such as
        `PVTO` / `PVTG`. When `None`, the table is flat (immiscible
        / simple-tabular).
    :param table_terminator: The string that separates tables within one
        keyword block (default `"/"`, i.e. every block is one table).
        Pass `"//"` for the double-slash convention used by some keywords.

    The parsed value is `List[List[Dict]]`; a list of tables (one per
    keyword occurrence / PVT region), each table being a list of row dicts.
    For miscible tables the `primary_key` field is duplicated into every
    inner-table row for convenience.
    """

    __slots__ = ("columns", "primary_key")

    def __init__(
        self,
        name: str,
        columns: typing.Sequence[Field[Number]],
        *,
        primary_key: str | None = None,
    ) -> None:
        super().__init__(name)
        self.columns: list[Field] = list(columns)
        self.primary_key = primary_key

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[PVTTable[Number]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        all_tables: list[PVTTable] = []
        for record in records:
            if self.primary_key is not None:
                tables = self._parse_miscible(record.body)
                all_tables.extend(tables)
            else:
                table = self._parse_flat(record.body)
                all_tables.append(table)

        return all_tables or None

    def _row_from_tokens(self, tokens: typing.Sequence[str]) -> PVTRow[Number]:
        """
        Convert a token list to a row dict using `self.columns`.

        Trailing optional columns that are absent from `tokens` fall back
        to their declared `default` values.  The Eclipse `1*` default
        designator is treated as absent (falls back to `default`).
        """
        row: PVTRow = {}
        n_tokens = len(tokens)
        for idx, column in enumerate(self.columns):
            raw = tokens[idx] if idx < n_tokens else None
            row[column.name] = column.parse(raw, self.name)
        return row

    def _parse_flat(self, body: str) -> PVTTable[Number]:
        """
        Parse a flat (immiscible/simple) table body.

        A `/`-delimited segment may hold exactly one row, or several rows
        packed together under a single closing `/` for the whole table - both
        conventions appear in real decks (`SWOF`/`SGOF` typically terminate
        each row individually; `PVDO`/`PVDG` typically terminate the whole
        table with one closing `/`, rows separated only by whitespace). Token
        counts are chunked into `n_columns`-sized rows to handle both, which
        is unambiguous here because every column in these tables is required
        (no optional trailing field to create row-width ambiguity).
        """
        table: PVTTable = []
        n_columns = len(self.columns)
        for segment in body.split("/"):
            tokens = tokenize(segment)
            if not tokens:
                continue

            n_tokens = len(tokens)
            if n_tokens % n_columns != 0:
                raise DeckParseError(
                    f"{self.name}: row segment has {n_tokens} token(s); "
                    f"expected a multiple of {n_columns} (one row per "
                    f"{n_columns} tokens)."
                )
            for start in range(0, n_tokens, n_columns):
                table.append(self._row_from_tokens(tokens[start : start + n_columns]))
        return table

    def _parse_miscible(self, body: str) -> list[PVTTable[Number]]:
        """
        Parse a miscible (PVTO/PVTG-style) keyword body.

        Each `/`-terminated segment holds either:

        - Zero or more *continuation* rows for the current primary-key group
        (`n_columns` tokens each) - e.g. several undersaturated pressure
        points sharing one closing `/`, or
        - A *new-group* introduction (`n_columns + 1` tokens: the primary-key
        value followed by its first row), optionally followed by more
        continuation rows for that same, just-introduced group, all still
        sharing that one closing `/`.

        The two shapes are distinguished by token count modulo `n_columns`
        (`0` -> pure continuation, `1` -> new group + continuations), which is
        unambiguous since every registered miscible table has `n_columns >= 2`.
        A segment with exactly one token is the degenerate new-group case with
        zero rows - a standalone primary-key line with no row data on it,
        which some decks use instead of the inline form.
        """
        tables: list[PVTTable] = []
        current_table: PVTTable = []
        current_pk_value: float | None = None
        pk_column_name = self.primary_key
        assert pk_column_name is not None  # guarded by caller
        n_columns = len(self.columns)
        assert n_columns >= 2, f"{self.name}: miscible tables need >= 2 data columns."

        for segment in body.split("/"):
            tokens = tokenize(segment)
            n_tokens = len(tokens)

            if n_tokens == 0:
                if current_table:
                    tables.append(current_table)
                    current_table = []
                    current_pk_value = None
                continue

            idx = 0
            if n_tokens % n_columns == 1:
                # New primary-key group: leading token introduces it.
                try:
                    current_pk_value = float(tokens[0])
                except ValueError as exc:
                    raise DeckParseError(
                        f"{self.name}: expected a numeric {pk_column_name!r} "
                        f"value, got {tokens[0]!r}."
                    ) from exc
                idx = 1
            elif n_tokens % n_columns != 0:
                raise DeckParseError(
                    f"{self.name}: row segment has {n_tokens} token(s); expected "
                    f"a multiple of {n_columns} (continuation rows), optionally "
                    f"prefixed by one {pk_column_name!r} value."
                )

            while idx < n_tokens:
                if current_pk_value is None:
                    raise DeckParseError(
                        f"{self.name}: data row encountered before any "
                        f"{pk_column_name!r} value was introduced."
                    )
                row = self._row_from_tokens(tokens[idx : idx + n_columns])
                row[pk_column_name] = current_pk_value
                current_table.append(row)
                idx += n_columns

        if current_table:
            tables.append(current_table)
        return tables


TimeUnit = typing.Literal["days", "hours", "seconds"]


def timedelta_to_timeunit(delta: datetime.timedelta, time_unit: TimeUnit) -> int:
    if time_unit in ("days", "seconds"):
        return getattr(delta, time_unit)
    return delta.days * c.HOURS_PER_DAY


def get_schedule_times(deck: Deck, time_unit: TimeUnit = "days") -> dict[int, float]:
    """
    One linear pass over `deck.records` (already in file order), mapping
    each record's `.start` offset to the elapsed-time clock value in effect
    at that point.

    `TSTEP` advances the clock by the sum of its step values.
    `DATES` jumps the clock to each listed calendar date in order; the
    record's `.start` is assigned the time of the **last** date in the
    block (since all subsequent records come after all those dates have
    elapsed).

    Both keywords' own records get the *new* clock value (they are the
    transition point). Records before the first `TSTEP`/`DATES` get `0.0`.

    :param deck: Scanned deck.
    :param time_unit: Time unit for elapsed time values (`"days"`,
        `"hours"`, or `"seconds"`). Defaults to `"days"`.
    :returns: `{record.start: elapsed_time}` for every record in `deck`.
    """
    # Resolve the simulation start date from `START` keyword, if present.
    # This makes sure `DATES`-based elapsed time is correct even when the first `DATES`
    # entry is not the same as `START`.
    start_record = deck.get_first_record_for("START")
    start_date: datetime.date | None = None
    if start_record is not None:
        start_body = start_record.body.split("/", 1)[0]
        start_tokens = tokenize(start_body, expand_repeats=True)
        if len(start_tokens) >= 3:
            try:
                start_date = _parse_date(start_tokens, "START")
            except DeckParseError:
                pass  # Malformed `START`. Fall back to lazy discovery

    times: dict[int, float] = {}
    current_time = 0.0

    for record in deck.records:
        if record.keyword == "TSTEP":
            body = record.body.split("/", 1)[0]
            step_sum = sum(float(token) for token in tokenize(body, expand_repeats=True) if token)
            current_time += step_sum
            times[record.start] = current_time

        elif record.keyword == "DATES":
            # A single `DATES` block may carry multiple date entries, each
            # terminated by its own '/'.  Walk every segment that has tokens.
            for segment in record.body.split("/"):
                tokens = tokenize(segment, expand_repeats=True)
                if not tokens:
                    continue
                try:
                    parsed_date = _parse_date(tokens, "DATES")
                except DeckParseError:
                    continue  # Skip malformed segments gracefully

                if start_date is None:
                    # First `DATES` seen with no `START`. We treat it as time-zero.
                    start_date = parsed_date
                    current_time = 0.0
                else:
                    delta = parsed_date - start_date
                    current_time = float(timedelta_to_timeunit(delta, time_unit))

            # Stamp the record with the time of the *last* date in the block.
            # All records that follow in the deck occur after every date listed
            # here has elapsed.
            times[record.start] = current_time

        else:
            times[record.start] = current_time
    return times


class ScheduledRecordKeyword(RepeatedRecordKeyword[T | float]):
    """
    A `RepeatedRecordKeyword` whose parsed records are also stamped with `"schedule_time"`

    `"schedule_time"` elapsed time in the deck's unit system (days/hours/seconds)
    since simulation start.

    Expects `schedule_times` (from `DeckFile._schedule_times`, computed once per
    file, same lifecycle as `operations`) to be passed in by the caller.
    Falls back to computing it fresh from `deck` if called directly without
    one, so this still works standalone, just less efficiently.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
        time_unit: TimeUnit = "days",
    ) -> list[dict[str, T | float | None]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        times = (
            schedule_times if schedule_times is not None else get_schedule_times(deck, time_unit)
        )
        results: list[dict[str, T | float | None]] = []
        for record in records:
            schedule_time = times.get(record.start, 0.0)
            for line in record.body.split("/"):
                tokens = tokenize(line, expand_repeats=True)
                if not tokens:
                    continue

                parsed = self.parse_tokens(tokens)
                parsed["schedule_time"] = schedule_time
                results.append(parsed)
        return results or None
