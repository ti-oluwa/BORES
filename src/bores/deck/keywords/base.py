"""
Base implementation for Eclipse keyword parsers, and concrete implementations
for various common keyword patterns.
"""

import abc
import datetime
import typing
from collections.abc import Collection

import attrs
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.datastructures import GridDimensions
from bores.deck.core import (
    Deck,
    DeckParseError,
    NumericRun,
    Record,
    first_segment,
    parse_numeric_runs,
    report,
    split_segments,
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
    "HeaderedScheduledRecordKeyword",
    "Keyword",
    "RecordKeyword",
    "RepeatedRecordKeyword",
    "ScheduledRecordKeyword",
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
    :param required: Whether this field must be present. Trailing optional
        fields fall back to `default` when the record has fewer tokens.
    :param default: Default value when the field is absent and
        `required=False`. Must be one of `options` when `options` is given.
    :param options: The complete set of values the field may take, compared
        against the value *after* `type` has converted it. When given, any other
        value is rejected with a `DeckParseError`.
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
                f"Default should be one of {', '.join(sorted(map(str, options)))}"
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
                f"{', '.join(sorted(map(str, self.options)))}."
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
        record = deck.get_single_record_for(self.name)
        if record is None:
            return None

        # `record.body` spans up to the next keyword line, since some
        # keywords need multiple internal `/` segments. We only
        # want the first (and only) `/`-terminated record here.
        body = first_segment(record.body)
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
            for line in split_segments(record.body):
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
        default_value: Number = 0.0,
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
            regardless of what was passed for `default_value`. Use for
            `MULTX`, `MULTY`, `MULTZ`, and their `-` variants.
        :param column_shape: Optional tuple of dimension-name strings declaring
            a valid "short form" token count for this keyword. Each string
            must be one of `"nx"`, `"ny"`, `"nz"`. When the token count
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
        source_states: dict[tuple[str, int], FloatArray[OneDimension]] | None = None,
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
        :param source_states: Arrays already resolved for `COPY` sources during this parse, by
            `(source name, number of that source's events replayed)`. Reading a source more than
            once, or through a chain of copies, then costs one replay. The arrays are only read.
        :returns: The resolved array, or `None` if this keyword has no
            data block and no operator ever targets it.
        """
        events = self._timeline(deck, dims, operations=operations)
        if stop_before_order is not None:
            events = [event for event in events if event[0] < stop_before_order]
        if not events:
            return None

        if source_states is None:
            source_states = {}
        array = np.full(dims.n_cells, self.default_value, dtype=np.float64)

        for _, kind, payload in events:
            if kind == "assign":
                self._assign(array, payload, dims, strict=deck.strict)
            else:  # kind == "operate"
                operation: Operation = payload
                apply_operation(
                    array,
                    operation,
                    dims,
                    resolve_source=lambda _name, _operation=operation: self._resolve_source(
                        deck, dims, _operation, operations, source_states
                    ),
                )
        return array.astype(self.dtype, copy=False)  # type: ignore[return-value]

    @staticmethod
    def _resolve_source(
        deck: Deck,
        dims: GridDimensions,
        operation: Operation,
        operations: list[Operation] | None,
        source_states: dict[tuple[str, int], FloatArray[OneDimension]],
    ) -> FloatArray[OneDimension] | None:
        """
        Resolve the source array of a `COPY` *as of* the point in the file where the `COPY`
        itself occurs, not as the source ends up after later events.

        :param deck: Scanned deck.
        :param dims: Resolved grid dimensions.
        :param operation: The `COPY` operation.
        :param operations: Every resolved operation in the deck.
        :param source_states: See `_resolve`.
        :returns: The source array (read-only, shared), or `None` if the source has no events
            before the `COPY`.
        """
        name = operation.source
        assert name is not None
        # The source's own default and short form, when known from the keyword registry. The name
        # based guess (multiplier arrays start at 1.0, others at 0.0) is only a fallback.
        default = operation.source_default
        if default is None:
            default = 1.0 if name.startswith("MULT") else 0.0
        probe = ArrayKeyword(
            name, default_value=default, column_shape=operation.source_column_shape
        )

        replayed = sum(
            1
            for event in probe._timeline(deck, dims, operations=operations)
            if event[0] < operation.order
        )
        if replayed == 0:
            return None

        key = (name, replayed)
        state = source_states.get(key)
        if state is None:
            state = probe._resolve(
                deck,
                dims,
                operations=operations,
                stop_before_order=operation.order,
                source_states=source_states,
            )
            assert state is not None
            state.flags.writeable = False
            source_states[key] = state
        return state

    def _assign(
        self,
        array: FloatArray[OneDimension],
        record: Record,
        dims: GridDimensions,
        *,
        strict: bool,
    ) -> None:
        """
        Write one explicit data block of this keyword into `array`.

        The block must hold one value per cell, or (when the keyword has a `column_shape`) one per
        column. A single value standing for the whole array is accepted unless `strict`. Defaulted
        (`N*`) entries cannot be given a value, so they are dropped from the block (which moves
        every value after them) with a warning, or rejected when `strict`.

        :param array: Flat `float64` working array, modified in place.
        :param record: The record holding the data block.
        :param dims: Resolved grid dimensions.
        :param strict: Raise instead of warning.
        :raises DeckParseError: If a value is not a number or the block has the wrong size.
        """
        runs = parse_numeric_runs(first_segment(record.body), name=self.name)

        defaulted = sum(run.entry_count for run in runs if run.values is None)
        if defaulted:
            report(
                f"{self.name}: record body contains bare 'N*' repeat(s) (empty-value tokens); "
                f"{defaulted} token(s) dropped, which moves every value after them.",
                strict=strict,
                stacklevel=4,
            )
            runs = [run for run in runs if run.values is not None]

        total_count = sum(run.entry_count for run in runs)
        if total_count == 0:
            report(
                f"{self.name}: assign event has zero tokens; skipping.",
                strict=strict,
                stacklevel=4,
            )
            return

        if total_count == dims.n_cells:
            fill_from_runs(array, runs)
            return

        # A short form (one value per column) is small by construction, so it is cheap to
        # expand before tiling it over the grid.
        short_count = self._short_form_count(dims) if self.column_shape is not None else None
        if short_count is not None and total_count == short_count:
            short_array = np.empty(total_count, dtype=np.float64)
            fill_from_runs(short_array, runs)
            array[:] = self._broadcast_short_form(short_array, dims)
            return

        if total_count == 1:
            # One value for the whole grid. Eclipse does not allow it (an array needs a value
            # per cell), but it is unambiguous, so it is accepted unless asked to be strict.
            if strict:
                raise DeckParseError(
                    f"{self.name} has a single value but the grid has {dims.n_cells} cells. "
                    "Write one value per cell, or `N*value` to repeat a value. "
                    "This is an error because the deck is being parsed with `strict=True`."
                )
            value = runs[0].values
            array[:] = value if isinstance(value, float) else float(value[0])  # type: ignore[index]
            return

        expected_desc = f"1 or {dims.n_cells}"
        if short_count is not None:
            expected_desc += f" or {short_count} (short form: {' * '.join(self.column_shape)})"  # type: ignore[arg-type]
        raise DeckParseError(f"{self.name} expected {expected_desc} value(s); got {total_count}.")

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
        keyword: `("assign", Record)` for each of its own explicit
        data-block records, interleaved with `("operate", Operation)`
        for every operator record that targets it.

        Both event kinds share one `order` namespace - a
        `(record_start, line_index)` tuple. Nothing is parsed here: a data block is only read when
        it is replayed.

        :returns: Events sorted ascending (earliest in file first).
        """
        events: list[tuple[tuple[int, int], str, typing.Any]] = [
            ((record.start, 0), "assign", record) for record in deck.get_records_for(self.name)
        ]

        if operations is None:
            operations = resolve_operations(deck, dims)

        for operation in operations:
            if operation.target == self.name:
                events.append((operation.order, "operate", operation))

        events.sort(key=lambda e: e[0])
        return events


def fill_from_runs(array: FloatArray[OneDimension], runs: typing.Iterable[NumericRun]) -> None:
    """
    Write `runs` (as read by `parse_numeric_runs`, with no defaulted runs) into `array`, one
    after another from its start.

    :param array: Flat array with room for every run. Modified in place.
    :param runs: The runs to write.
    """
    position = 0
    for count, values in runs:
        array[position : position + count] = values
        position += count


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


def _parse_time_of_day(token: str, name: str) -> datetime.timedelta:
    """
    Parse an Eclipse time of day `HH:MM:SS` or `HH:MM:SS.SSS` into the time elapsed since
    midnight.

    :param token: The time token.
    :param name: Keyword name for error messages.
    :returns: Time since midnight.
    :raises DeckParseError: If the token is not a valid time of day.
    """
    parts = token.split(":")
    if len(parts) != 3:
        raise DeckParseError(
            f"{name}: time {token!r} is not of the form HH:MM:SS or HH:MM:SS.SSS."
        )
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError as exc:
        raise DeckParseError(f"{name}: time {token!r} is not numeric.") from exc
    if not (0 <= hours <= 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise DeckParseError(f"{name}: time {token!r} is out of range.")
    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _parse_datetime(tokens: typing.Sequence[str], name: str) -> datetime.datetime:
    """
    Parse an Eclipse date with an optional time of day, `[day, month, year]` or
    `[day, month, year, time]`, into a `datetime.datetime`. A date without a time is midnight.

    :param tokens: Three or four string tokens: day, month abbreviation, year and, optionally,
        a time of day such as `12:00:00`.
    :param name: Keyword name for error messages.
    :returns: Parsed date and time.
    :raises DeckParseError: If the tokens cannot be parsed.
    """
    date = _parse_date(tokens, name)
    moment = datetime.datetime(date.year, date.month, date.day)
    if len(tokens) >= 4:
        moment += _parse_time_of_day(tokens[3], name)
    return moment


class FlagKeyword(Keyword[bool]):
    """
    A bare presence-only switch keyword that carries no data section.

    Examples: `OIL`, `GAS`, `WATER`, `DISGAS`, `VAPOIL`,
    `FIELD`, `METRIC`, `LAB`, `NOSIM`, `UNIFIN`, `UNIFOUT`.

    `parse` returns `True` when the keyword appears anywhere in
    the deck, `False` otherwise. It never returns `None` so callers
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


class OptionListKeyword(Keyword[list[str]]):
    """
    A keyword whose one record is a free-length list of option mnemonics (`SATOPTS`, `EQLOPTS`).

    Which options are switched on is the whole content, so `parse` returns them as a list of
    upper-case names in the order written, e.g. `["HYSTER"]` for `SATOPTS HYSTER /`. Defaulted
    (`1*`) positions switch nothing on and are left out. The keyword being present with no options
    gives `[]`, so it stays distinguishable from `None`, the keyword being absent.

    The names are not checked against a list: the options an Eclipse version accepts differ, and
    reading a deck is not the place to reject one.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[str] | None:
        record = deck.get_single_record_for(self.name)
        if record is None:
            return None
        return [token.upper() for token in tokenize(first_segment(record.body)) if token != "1*"]


class DateKeyword(Keyword[datetime.datetime]):
    """
    A keyword whose single record is an Eclipse-format date
    `D MON YYYY /`, optionally followed by a time of day, `D MON YYYY HH:MM:SS /`.

    Used for `START`.

    `parse` returns a `datetime.datetime` (midnight if the deck gives no time), or `None`
    when the keyword is absent.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> datetime.datetime | None:
        record = deck.get_single_record_for(self.name)
        if record is None:
            return None

        body = first_segment(record.body)
        tokens = tokenize(body)
        return _parse_datetime(tokens, self.name)


class DatesKeyword(Keyword[list[datetime.datetime]]):
    """
    The `DATES` keyword: a sequence of one or more `/`-terminated
    date entries inside one `DATES … /` block, each of the form
    `D MON YYYY /` or `D MON YYYY HH:MM:SS /`.

    Multiple `DATES` blocks in the same deck are concatenated in file
    order and returned as a single flat list.

    `parse` returns a list of `datetime.datetime` objects (midnight for an entry with no time),
    or `None` when the keyword is absent.

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
    ) -> list[datetime.datetime] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        dates: list[datetime.datetime] = []
        for record in records:
            for segment in split_segments(record.body):
                tokens = tokenize(segment)
                if not tokens:
                    continue
                dates.append(_parse_datetime(tokens, self.name))
        return dates or None


TableRow = dict[str, T]
"""One row of a PVT/saturation/... table: a `{column_name: value}` dict."""

Table = list[TableRow[T]]
"""One saturation/PVT/... table: a list of row dicts in ascending primary-key order."""


class TableKeyword(Keyword[list[Table[Number]]]):
    """
    A keyword whose body contains tabulated data, one `/`-terminated segment per region.

    This covers two sub-patterns:

    **Simple (immiscible) tables**: each non-empty `/`-terminated segment is one region's table,
    and the table is the rows packed in that segment, `n_columns` values per row. The parsed value
    has one entry per region, in order, so entry `i` belongs to region `i + 1`. That holds for
    both layouts these keywords use:

    - one table per region, rows separated only by whitespace and the whole table closed by one
      `/` (`SWOF`, `SGOF`, `SWFN`, `SGFN`, `SOF2`, `SOF3`, `PVDG`, `PVDO`, `RSVD`, ...), and
    - one row per region, each closed by its own `/` (`DENSITY`, `PVTW`, `ROCK`), which is the
      same thing with a single-row table.

    **Compact style**: rows that are each closed by their own `/`, with the table closed by a
    lone `/` after them (`SWOF\n 0.2 0 1 0 /\n 1 1 0 0 /\n/`), are read as the rows of one
    table. That is only done when the body ends in a lone `/` and every segment before it holds
    exactly one row; anything else is read one table per segment.

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

    The parsed value is `List[List[Dict]]`; a list of tables (one per region, in order across the
    keyword occurrences), each table being a list of row dicts.
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
    ) -> list[Table[Number]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        all_tables: list[Table] = []
        for record in records:
            if self.primary_key is not None:
                tables = self._parse_miscible(record.body)
                all_tables.extend(tables)
            else:
                all_tables.extend(self._parse_flat_tables(record.body))

        return all_tables or None

    def _row_from_tokens(self, tokens: typing.Sequence[str]) -> TableRow[Number]:
        """
        Convert a token list to a row dict using `self.columns`.

        Trailing optional columns that are absent from `tokens` fall back
        to their declared `default` values. The Eclipse `1*` default
        designator is treated as absent (falls back to `default`).
        """
        return typing.cast(TableRow[Number], parse_tokens(self.name, self.columns, tokens))

    def _parse_flat_tables(self, body: str) -> list[Table[Number]]:
        """
        Parse a flat (immiscible/simple) keyword body into one table per region.

        Each non-empty `/`-delimited segment is one table: its tokens are chunked into
        `n_columns`-sized rows, which is unambiguous because every column in these tables is
        required (no optional trailing field to create row-width ambiguity). The one exception is
        the compact style, in which every segment is a single row and a lone `/` closes the
        whole: those rows are one table (see the class documentation).

        :param body: Keyword body.
        :returns: The tables, in order. Segments without data are skipped.
        :raises DeckParseError: If a segment does not hold a whole number of rows.
        """
        n_columns = len(self.columns)
        segments = split_segments(body)
        token_lists = [tokenize(segment) for segment in segments]
        for tokens in token_lists:
            if len(tokens) % n_columns != 0:
                raise DeckParseError(
                    f"{self.name}: row segment has {len(tokens)} token(s); "
                    f"expected a multiple of {n_columns} (one row per "
                    f"{n_columns} tokens)."
                )

        data = [tokens for tokens in token_lists if tokens]
        # `[..., rows, "", remainder]`: the segment before the remainder is the lone `/`.
        closed_by_lone_slash = len(token_lists) >= 2 and not token_lists[-2]
        if (
            closed_by_lone_slash
            and len(data) >= 2
            and all(len(tokens) == n_columns for tokens in data)
        ):
            data = [[token for tokens in data for token in tokens]]

        return [
            [
                self._row_from_tokens(tokens[start : start + n_columns])
                for start in range(0, len(tokens), n_columns)
            ]
            for tokens in data
        ]

    def _parse_miscible(self, body: str) -> list[Table[Number]]:
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
        tables: list[Table] = []
        current_table: Table = []
        current_pk_value: float | None = None
        pk_column_name = self.primary_key
        assert pk_column_name is not None  # guarded by caller
        n_columns = len(self.columns)
        assert n_columns >= 2, f"{self.name}: miscible tables need >= 2 data columns."

        for segment in split_segments(body):
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


def timedelta_to_timeunit(delta: datetime.timedelta, time_unit: TimeUnit) -> float:
    """
    Convert the difference between two deck dates, including any time of day, to `time_unit`.

    :param delta: Difference between two deck dates.
    :param time_unit: Unit to express the difference in.
    :returns: The difference in `time_unit`.
    """
    seconds = delta.total_seconds()
    if time_unit == "seconds":
        return seconds
    days = seconds / c.SECONDS_PER_DAY
    if time_unit == "days":
        return days
    return days * c.HOURS_PER_DAY


def get_schedule_clock(
    deck: Deck, time_unit: TimeUnit = "days"
) -> tuple[dict[int, float], list[float]]:
    """
    Takes one linear pass over `deck.records` (already in file order), tracking the elapsed-time
    clock that `TSTEP` and `DATES` advance.

    `TSTEP` advances the clock by each of its entries in turn, and every entry is a report time.
    `DATES` jumps the clock to each listed calendar date in order, and every date is a report time.

    Each record's `.start` offset is mapped to the clock value in effect at that point. A `TSTEP`
    or `DATES` record gets the clock value after its last entry, since every record that follows
    comes after all of them. Records before the first `TSTEP` or `DATES` get `0.0`.

    A `TSTEP` entry that is not a number always raises. Other problems are reported with
    `deck.strict` deciding between a warning and an error: a `START` that cannot be read (the
    first `DATES` is then time zero), a `DATES` entry that is not a date (it is skipped), and a
    `TSTEP` step or `DATES` entry that would move the clock backwards.

    :param deck: Scanned deck.
    :param time_unit: Time unit for elapsed time values (`"days"`, `"hours"`, or `"seconds"`).
        `TSTEP` entries are read in the deck's own time unit, which this must match.
    :returns: `({record.start: elapsed_time}, report_times)`. `report_times` is sorted, has no
        repeats, and always starts with `0.0`.
    :raises DeckParseError: If a `TSTEP` entry is not a number, or (when `deck.strict`) for any of
        the problems above.
    """
    # Resolve the simulation start date from `START` keyword, if present.
    # This makes sure `DATES`-based elapsed time is correct even when the first `DATES`
    # entry is not the same as `START`.
    start_record = deck.get_first_record_for("START")
    start_date: datetime.datetime | None = None
    if start_record is not None:
        start_body = first_segment(start_record.body)
        start_tokens = tokenize(start_body, expand_repeats=True)
        try:
            start_date = _parse_datetime(start_tokens, "START")
        except DeckParseError as exc:
            # Malformed `START`. Fall back to lazy discovery
            report(
                f"`START` cannot be read ({exc}). The first `DATES` entry is taken as time zero.",
                strict=deck.strict,
                stacklevel=3,
            )

    times: dict[int, float] = {}
    report_times: set[float] = {0.0}
    current_time = 0.0

    for record in deck.records:
        if record.keyword == "TSTEP":
            body = first_segment(record.body)
            for token in tokenize(body, expand_repeats=True):
                if token:
                    try:
                        step = float(token)
                    except ValueError:
                        hint = (
                            " A defaulted ('N*') step has no value." if token.endswith("*") else ""
                        )
                        raise DeckParseError(
                            f"TSTEP: non-numeric time-step value {token!r}.{hint}"
                        ) from None
                    if step < 0:
                        report(
                            f"`TSTEP` step {token} moves the clock backwards.",
                            strict=deck.strict,
                            stacklevel=3,
                        )
                    current_time += step
                    report_times.add(current_time)
            times[record.start] = current_time

        elif record.keyword == "DATES":
            # A single `DATES` block may carry multiple date entries, each
            # terminated by its own '/'. Walk every segment that has tokens.
            for segment in split_segments(record.body):
                tokens = tokenize(segment, expand_repeats=True)
                if not tokens:
                    continue
                try:
                    parsed_date = _parse_datetime(tokens, "DATES")
                except DeckParseError as exc:
                    report(
                        f"`DATES` entry {tokens!r} cannot be read ({exc}). It is skipped.",
                        strict=deck.strict,
                        stacklevel=3,
                    )
                    continue

                if start_date is None:
                    # First `DATES` seen with no `START`. We treat it as time-zero.
                    start_date = parsed_date
                    current_time = 0.0
                else:
                    delta = parsed_date - start_date
                    previous_time = current_time
                    current_time = timedelta_to_timeunit(delta, time_unit)
                    if current_time < previous_time:
                        report(
                            f"`DATES` entry {' '.join(tokens)} is earlier than the deck's clock "
                            f"already reached. It is read as an earlier time.",
                            strict=deck.strict,
                            stacklevel=3,
                        )
                report_times.add(current_time)

            times[record.start] = current_time

        else:
            times[record.start] = current_time
    return times, sorted(report_times)


def get_schedule_times(deck: Deck, time_unit: TimeUnit = "days") -> dict[int, float]:
    """
    Maps each record's `.start` offset to the elapsed-time clock value in effect at that point.
    See `get_schedule_clock` for how `TSTEP` and `DATES` advance the clock.

    :param deck: Scanned deck.
    :param time_unit: Time unit for elapsed time values (`"days"`, `"hours"`, or `"seconds"`).
    :returns: `{record.start: elapsed_time}` for every record in `deck`.
    """
    return get_schedule_clock(deck, time_unit)[0]


def get_report_times(deck: Deck, time_unit: TimeUnit = "days") -> list[float]:
    """
    Every report time a deck asks for: each entry of every `TSTEP` and each date of every
    `DATES`, as elapsed time since the start, and `0.0`. A `TSTEP 5*30 /` gives a report time
    after each of its five 30 day entries, not only after their sum.

    :param deck: Scanned deck.
    :param time_unit: Time unit for elapsed time values (`"days"`, `"hours"`, or `"seconds"`).
    :returns: Sorted report times without repeats, starting with `0.0`.
    """
    return get_schedule_clock(deck, time_unit)[1]


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
            for line in split_segments(record.body):
                tokens = tokenize(line, expand_repeats=True)
                if not tokens:
                    continue

                parsed = self.parse_tokens(tokens)
                parsed["schedule_time"] = schedule_time
                results.append(parsed)
        return results or None


class HeaderedScheduledRecordKeyword(Keyword[list[dict[str, typing.Any]]]):
    """
    A keyword whose body is one header record followed by zero or more
    detail records with a different field layout (`WELSEGS`, `COMPSEGS`),
    each occurrence stamped with `"schedule_time"` the same way
    `ScheduledRecordKeyword` stamps a flat `RepeatedRecordKeyword`.

    One dict per keyword occurrence in the deck, not one per detail
    record: `{**header_fields, "schedule_time": ..., "details": [{**detail_fields}, ...]}`.
    """

    __slots__ = ("detail_fields", "header_fields")

    def __init__(
        self,
        name: str,
        *,
        header_fields: typing.Sequence[Field[typing.Any]],
        detail_fields: typing.Sequence[Field[typing.Any]],
    ) -> None:
        super().__init__(name)
        self.header_fields = list(header_fields)
        self.detail_fields = list(detail_fields)

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
        time_unit: TimeUnit = "days",
    ) -> list[dict[str, typing.Any]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        times = (
            schedule_times if schedule_times is not None else get_schedule_times(deck, time_unit)
        )
        results: list[dict[str, typing.Any]] = []
        for record in records:
            lines = [tokenize(line) for line in split_segments(record.body)]
            lines = [line for line in lines if line]
            if not lines:
                continue

            parsed = parse_tokens(self.name, self.header_fields, lines[0])
            parsed["schedule_time"] = times.get(record.start, 0.0)
            parsed["details"] = [
                parse_tokens(self.name, self.detail_fields, line) for line in lines[1:]
            ]
            results.append(parsed)
        return results or None


class SettingsKeyword(Keyword[list[dict[str, typing.Any]]]):
    """
    A keyword that may appear many times in the `SCHEDULE` section, each occurrence holding a list
    of `NAME=VALUE` settings (`ZIPPY2`) that take effect from its `schedule_time` onwards.

    `parse` returns one dict per occurrence, in file order:

    - `"settings"`: `{NAME: value_or_None}`. `NAME` is upper-cased. The value is the text after the
      `=`, exactly as written (so it is for the caller to turn into a number), or `None` for a name
      written without `=value`. A later setting of the same name replaces an earlier one.
    - `"schedule_time"`: elapsed time at which the occurrence takes effect, in the deck's time
      unit.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
        time_unit: TimeUnit = "days",
    ) -> list[dict[str, typing.Any]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        times = (
            schedule_times if schedule_times is not None else get_schedule_times(deck, time_unit)
        )
        results: list[dict[str, typing.Any]] = []
        for record in records:
            settings: dict[str, str | None] = {}
            for token in tokenize(first_segment(record.body)):
                if token == "1*":
                    continue
                name, equals, value = token.partition("=")
                settings[name.strip().upper()] = value.strip() if equals else None
            results.append({"settings": settings, "schedule_time": times.get(record.start, 0.0)})
        return results


class MultiRecordScheduledKeyword(Keyword[list[dict[str, typing.Any]]]):
    """
    A keyword that may appear many times in the `SCHEDULE` section, each occurrence being a fixed
    sequence of differently laid out `/`-terminated records (`TUNING`).

    Every record is positional and every item optional, so an empty record (a lone `/`) is the way
    to leave the whole record at the simulator's defaults.

    `parse` returns one dict per occurrence, in file order: one entry per record, keyed by the
    record's name and holding that record's fields, plus `"schedule_time"` (elapsed time at which
    the occurrence takes effect, in the deck's time unit).

    An occurrence with fewer terminated records than the keyword has is reported (a warning, or a
    `DeckParseError` when `deck.strict`), and the records it lacks are read as empty. Anything
    after the last record that is not a lone `/` is reported the same way and ignored.
    """

    __slots__ = ("records",)

    def __init__(
        self,
        name: str,
        records: typing.Sequence[tuple[str, typing.Sequence[Field[typing.Any]]]],
    ) -> None:
        """
        :param name: Keyword name.
        :param records: `(record name, fields)` for each record, in the order they are written.
        """
        super().__init__(name)
        self.records = [(record_name, list(fields)) for record_name, fields in records]

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
        time_unit: TimeUnit = "days",
    ) -> list[dict[str, typing.Any]] | None:
        occurrences = deck.get_records_for(self.name)
        if not occurrences:
            return None

        times = (
            schedule_times if schedule_times is not None else get_schedule_times(deck, time_unit)
        )
        results: list[dict[str, typing.Any]] = []
        for occurrence in occurrences:
            # Everything before a `/` is a record; what follows the last `/` is not.
            segments = split_segments(occurrence.body)[:-1]
            if len(segments) < len(self.records):
                report(
                    f"{self.name} has {len(segments)} `/`-terminated record(s) but needs "
                    f"{len(self.records)}. The missing record(s) are read as empty, so every item "
                    "in them takes its default.",
                    strict=deck.strict,
                    stacklevel=3,
                )
            extra = [segment for segment in segments[len(self.records) :] if segment.strip()]
            if extra:
                report(
                    f"{self.name} has data after its {len(self.records)} records, which is "
                    "ignored.",
                    strict=deck.strict,
                    stacklevel=3,
                )

            parsed: dict[str, typing.Any] = {}
            for index, (record_name, fields) in enumerate(self.records):
                segment = segments[index] if index < len(segments) else ""
                parsed[record_name] = parse_tokens(
                    self.name, fields, tokenize(segment, expand_repeats=True)
                )
            parsed["schedule_time"] = times.get(occurrence.start, 0.0)
            results.append(parsed)
        return results
