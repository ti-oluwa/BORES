"""
Eclipse `BOX` / `ENDBOX` / `EQUALS` / `ADD` / `MULTIPLY` /
`COPY` / `MAXVALUE` / `MINVALUE` operator resolution.

Operators in Eclipse allow bulk modification of per-cell arrays without
re-stating the full array. They work in file order: every
`EQUALS`/`ADD`/`MULTIPLY`/`COPY`/`MAXVALUE`/`MINVALUE`
record targets a named keyword array and is applied to a sub-grid box.
The active box is set by the most recent `BOX` record in file order
(or reset to the full grid by `ENDBOX`); individual operator records
may also carry an inline box override as six trailing integers.
"""

import typing
import warnings

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, first_segment, report, split_segments, tokenize
from bores.types import FloatArray, Integer, Number, OneDimension, ThreeDimensions

__all__ = [
    "OPERATOR_CONTROL_KEYWORDS",
    "Operation",
    "apply_operation",
    "resolve_operations",
]

BOX_KEYWORD = "BOX"
ENDBOX_KEYWORD = "ENDBOX"
OPERATOR_KEYWORDS: frozenset[str] = frozenset({
    "EQUALS",
    "ADD",
    "MULTIPLY",
    "COPY",
    "MAXVALUE",
    "MINVALUE",
})

OPERATOR_CONTROL_KEYWORDS: frozenset[str] = (
    frozenset({BOX_KEYWORD, ENDBOX_KEYWORD}) | OPERATOR_KEYWORDS
)
"""
Keywords that scope or drive operators rather than holding their own
per-cell data; never matched as a plain `bores.deck.keywords.base.ArrayKeyword` value.
"""


class Operation(typing.NamedTuple):
    """One resolved operator instruction (`EQUALS` / `ADD` / etc.)."""

    op: str
    """
    Operator name: one of `EQUALS`, `ADD`, `MULTIPLY`, `COPY`,
    `MAXVALUE`, `MINVALUE`.
    """

    target: str
    """Upper-cased name of the keyword array being modified."""

    value: Number | None
    """
    Scalar operand for `EQUALS` / `ADD` / `MULTIPLY` / `MAXVALUE` /
    `MINVALUE`; `None` for `COPY`.
    """

    source: str | None
    """Upper-cased source keyword name for `COPY`; `None` otherwise."""

    box: tuple[Integer, Integer, Integer, Integer, Integer, Integer]
    """**0-based** inclusive `(i1, i2, j1, j2, k1, k2)` sub-grid box."""

    order: tuple[int, int]
    """
    `(record_start, line_index)` sort key.

    `record_start` is the character offset of the operator keyword's own
    `bores.deck.core.Record` in the source text (same offset space
    used by data-block assign events), and `line_index` breaks ties between
    multiple operation lines within the same operator block.
    """

    source_default: Number | None = None
    """
    For `COPY`: the value cells of the source array have before anything is written to them (the
    source keyword's own default, e.g. `1.0` for `NTG`). `None` when the source is not a known
    array keyword, in which case a name-based guess is used.
    """

    source_column_shape: tuple[str, ...] | None = None
    """For `COPY`: the short-form `column_shape` of the source array keyword, if it has one."""


ArrayInfo = typing.Mapping[str, tuple[Number, tuple[str, ...] | None]]
"""
What the operator resolver needs to know about array keywords, by name: `(default_value,
column_shape)`. Built by `bores.deck.file.DeckFile` from its registry.
"""


def get_default_box(dims: GridDimensions) -> tuple[int, int, int, int, int, int]:
    """Return the 0-based inclusive box covering every cell in the grid."""
    return (0, dims.nx - 1, 0, dims.ny - 1, 0, dims.nz - 1)


def clamp_box(
    i1: Integer,
    i2: Integer,
    j1: Integer,
    j2: Integer,
    k1: Integer,
    k2: Integer,
    dims: GridDimensions,
) -> tuple[Integer, Integer, Integer, Integer, Integer, Integer]:
    """Clamp a raw (possibly out-of-bounds) box to grid extents and sort each pair."""
    i1, i2 = sorted((max(0, min(i1, dims.nx - 1)), max(0, min(i2, dims.nx - 1))))
    j1, j2 = sorted((max(0, min(j1, dims.ny - 1)), max(0, min(j2, dims.ny - 1))))
    k1, k2 = sorted((max(0, min(k1, dims.nz - 1)), max(0, min(k2, dims.nz - 1))))
    return i1, i2, j1, j2, k1, k2


def parse_box(
    tokens: typing.Sequence[str],
    fallback: tuple[Integer, Integer, Integer, Integer, Integer, Integer],
    dims: GridDimensions,
    *,
    label: str,
    strict: bool,
) -> tuple[Integer, Integer, Integer, Integer, Integer, Integer] | None:
    """
    Turn up to six 1-based box tokens (`I1 I2 J1 J2 K1 K2`) into a 0-based inclusive box.

    As in any Eclipse record, items left off the end are defaulted, and a bound written `1*` is
    defaulted too. A defaulted bound takes the matching bound of `fallback`.

    A box that does not fit the grid (a bound outside it, or a lower bound above its upper bound)
    is reported, and in a lenient parse is then clamped to the grid and sorted.

    :param tokens: Box tokens, as many as the record gave (extra tokens are ignored).
    :param fallback: 0-based box that supplies defaulted bounds.
    :param dims: Grid extent.
    :param label: What the box belongs to, for messages (e.g. "`BOX` record").
    :param strict: Raise instead of warning.
    :returns: The box, or `None` if a bound is not an integer (reported first).
    """
    padded = [*tokens[:6], *["1*"] * (6 - len(tokens[:6]))]
    bounds: list[Integer] = []
    for token, default in zip(padded, fallback, strict=True):
        if token == "1*":
            bounds.append(default)
            continue
        try:
            # Minus 1, to move from 1-based to 0-based indexing used internally.
            bounds.append(int(token) - 1)
        except ValueError:
            report(
                f"{label} has the non-integer box bound {token!r} (expected I1 I2 J1 J2 K1 K2).",
                strict=strict,
                stacklevel=3,
            )
            return None

    problems: list[str] = []
    for axis, (lower, upper), size in zip(
        "IJK",
        zip(bounds[0::2], bounds[1::2], strict=True),
        (dims.nx, dims.ny, dims.nz),
        strict=True,
    ):
        if lower < 0 or upper < 0 or lower >= size or upper >= size:
            problems.append(f"{axis} bounds {lower + 1}-{upper + 1} fall outside 1-{size}")
        elif lower > upper:
            problems.append(f"{axis}1 ({lower + 1}) is above {axis}2 ({upper + 1})")

    if problems:
        report(
            f"{label} does not fit the {dims.nx}x{dims.ny}x{dims.nz} grid: {'; '.join(problems)}. "
            "The box is clamped to the grid and its bounds sorted.",
            strict=strict,
            stacklevel=3,
        )
    return clamp_box(*bounds, dims=dims)


def resolve_operations(
    deck: Deck, dims: GridDimensions, *, array_info: ArrayInfo | None = None
) -> list[Operation]:
    """
    Walk every `BOX` / `ENDBOX` / operator record in deck order and
    resolve each operator record to a concrete `Operation`.

    `BOX` updates the active box for all subsequent operator records until
    `ENDBOX` resets it to the full grid extent. The active box is
    deck-order state, independent of which keyword each operator record
    targets.

    :param deck: Pre-scanned `bores.deck.core.Deck`.
    :param dims: Grid extent, needed to clamp/resolve box bounds and the
        default (whole-grid) box.
    :param array_info: Default value and short form of known array keywords, by name. Lets a
        `COPY` read its source array the way the source keyword itself would.
    :returns: Resolved operations in file order (by `order`).
    """
    default_box = get_default_box(dims)
    current_box = default_box
    operations: list[Operation] = []

    for record in deck.records:
        if record.keyword == BOX_KEYWORD:
            # A box that cannot be read must not be dropped quietly: the operators after it
            # would then act on the previous box (usually the whole grid).
            box = parse_box(
                tokenize(first_segment(record.body)),
                default_box,
                dims,
                label="`BOX` record",
                strict=deck.strict,
            )
            if box is None:
                warn_box_ignored(deck.strict)
                continue
            current_box = box

        elif record.keyword == ENDBOX_KEYWORD:
            current_box = default_box

        elif record.keyword in OPERATOR_KEYWORDS:
            for line_index, op in enumerate(
                parse_operator_records(
                    record.body,
                    record.keyword,
                    current_box,
                    dims,
                    strict=deck.strict,
                    array_info=array_info,
                )
            ):
                operations.append(op._replace(order=(record.start, line_index)))

    return operations


def warn_box_ignored(strict: bool) -> None:
    """Warn that an unreadable `BOX` was ignored and the previous box stays in force."""
    if not strict:
        report(
            "The unreadable `BOX` record is ignored, so the operators after it apply to the "
            "previous box.",
            strict=False,
            stacklevel=3,
        )


def parse_operator_records(
    body: str,
    op: str,
    box: tuple[Integer, Integer, Integer, Integer, Integer, Integer],
    dims: GridDimensions,
    *,
    strict: bool = False,
    array_info: ArrayInfo | None = None,
) -> typing.Iterator[Operation]:
    """
    Parse the (possibly multi-record) body of a single operator block.

    Each `/`-delimited line is one record. An optional trailing 6
    integers on the same record override the ambient `BOX` for that
    record only (standard Eclipse shorthand). The per-record override is
    clamped to grid bounds, just like an explicit `BOX` keyword.

    :param body: Raw record body (between the operator keyword and its
        terminating `/`), which may itself contain embedded `/`
        characters separating individual records.
    :param op: Operator name.
    :param box: Ambient box active at this point in the deck. A bound written as
        `1*` in a record's own box takes the matching bound from this box.
    :param dims: Grid dimensions, used to clamp per-record box overrides.
    :param strict: Raise a `DeckParseError` for a malformed record (or box) instead of skipping it
        with a warning.
    :param array_info: See `resolve_operations`.
    :yields: One `Operation` per record line (`order` field is
        a placeholder `(0, 0)` — the caller replaces it).
    """
    for line in split_segments(body):
        tokens = tokenize(line)
        if not tokens:
            continue

        record_box = box

        if op == "COPY":
            if len(tokens) < 2:
                report(
                    f"`COPY` record {tokens!r} needs a source and target keyword; skipping it.",
                    strict=strict,
                    stacklevel=3,
                )
                continue
            source, target = tokens[0].upper(), tokens[1].upper()
            value = None
            extra = tokens[2:]
        else:
            if len(tokens) < 2:
                report(
                    f"`{op}` record {tokens!r} needs a target keyword and value; skipping it.",
                    strict=strict,
                    stacklevel=3,
                )
                continue
            target = tokens[0].upper()
            source = None
            try:
                value = float(tokens[1])
            except ValueError:
                report(
                    f"`{op}` record for {target!r} has non-numeric value {tokens[1]!r}; "
                    "skipping it.",
                    strict=strict,
                    stacklevel=3,
                )
                continue
            extra = tokens[2:]

        if extra:
            # A defaulted bound (`1*`, or one left off the end) keeps the ambient box's own bound.
            parsed = parse_box(
                extra,
                box,
                dims,
                label=f"`{op}` record for {target!r}",
                strict=strict,
            )
            # An unreadable inline box was reported above; the record then uses the ambient box.
            if parsed is not None:
                record_box = parsed

        source_default, source_column_shape = (
            (array_info or {}).get(source, (None, None)) if source is not None else (None, None)
        )
        yield Operation(
            op=op,
            target=target,
            value=value,
            source=source,
            box=record_box,
            order=(0, 0),
            source_default=source_default,
            source_column_shape=source_column_shape,
        )


def get_box_view(
    array: FloatArray[OneDimension],
    box: tuple[Integer, Integer, Integer, Integer, Integer, Integer],
    dims: GridDimensions,
) -> FloatArray[ThreeDimensions]:
    """
    Return the cells of `box` as a writable 3-D view of the flat per-cell `array`.

    A flat array in Eclipse order (`i` fastest, then `j`, then `k`) is a `(nz, ny, nx)` array in
    C order, so a box is a plain slice of it. Working on that view needs no index array and
    copies nothing, however large the box: a full-grid operator is the common case, and an
    index array for it would be as large as the grid itself.

    :param array: Flat `(n_cells,)` array. It must be contiguous, as every array built here is.
    :param box: **0-based** inclusive `(i1, i2, j1, j2, k1, k2)` box inside the grid.
    :param dims: Grid extent.
    :returns: A view of `array` over the box, indexed `[k, j, i]`.
    """
    i1, i2, j1, j2, k1, k2 = box
    grid = array.reshape(dims.nz, dims.ny, dims.nx)
    return typing.cast("FloatArray[ThreeDimensions]", grid[k1 : k2 + 1, j1 : j2 + 1, i1 : i2 + 1])


def apply_operation(
    array: FloatArray[OneDimension],
    operation: Operation,
    dims: GridDimensions,
    resolve_source: typing.Callable[[str], FloatArray[OneDimension] | None],
) -> None:
    """
    Apply one resolved `Operation` to `array` in place, over its box.

    :param array: Flat `(n_cells,)` array for the operation's target
        keyword (already validated to belong to that keyword).
    :param operation: The resolved operation to apply.
    :param dims: Grid extent (for IJK -> flat index mapping).
    :param resolve_source: Callback returning the current array for a given
        keyword name, used by `COPY` to fetch the source array. May
        return `None` if the source keyword has no data yet. The returned array is only read.
    """
    cells = get_box_view(array, operation.box, dims)

    if operation.op == "EQUALS":
        cells[...] = operation.value
    elif operation.op == "ADD":
        cells += operation.value  # type: ignore
    elif operation.op == "MULTIPLY":
        cells *= operation.value  # type: ignore
    elif operation.op == "MAXVALUE":
        np.maximum(cells, operation.value, out=cells)  # type: ignore
    elif operation.op == "MINVALUE":
        np.minimum(cells, operation.value, out=cells)  # type: ignore
    elif operation.op == "COPY":
        assert operation.source is not None
        source_array = resolve_source(operation.source)
        if source_array is None:
            warnings.warn(
                f"`COPY` source {operation.source!r} has no data; skipping "
                f"`COPY` into {operation.target!r}.",
                stacklevel=2,
            )
            return
        cells[...] = get_box_view(source_array, operation.box, dims)
