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
from bores.deck.core import Deck, tokenize
from bores.types import FloatArray, IntArray, OneDimension

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
per-cell data; never matched as a plain
`bores.deck.keywords.array.ArrayKeyword` value.
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

    value: float | None
    """
    Scalar operand for `EQUALS` / `ADD` / `MULTIPLY` / `MAXVALUE` /
    `MINVALUE`; `None` for `COPY`.
    """

    source: str | None
    """Upper-cased source keyword name for `COPY`; `None` otherwise."""

    box: tuple[int, int, int, int, int, int]
    """**0-based** inclusive `(i1, i2, j1, j2, k1, k2)` sub-grid box."""

    order: tuple[int, int]
    """
    `(record_start, line_index)` sort key.

    `record_start` is the character offset of the operator keyword's own
    `bores.deck.core.Record` in the source text (same offset space
    used by data-block assign events), and `line_index` breaks ties between
    multiple operation lines within the same operator block.
    """


def get_default_box(dims: GridDimensions) -> tuple[int, int, int, int, int, int]:
    return (0, dims.nx - 1, 0, dims.ny - 1, 0, dims.nz - 1)


def clamp_box(
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    k1: int,
    k2: int,
    dims: GridDimensions,
) -> tuple[int, int, int, int, int, int]:
    """Clamp a raw (possibly out-of-bounds) box to grid extents and sort each pair."""
    i1, i2 = sorted((max(0, min(i1, dims.nx - 1)), max(0, min(i2, dims.nx - 1))))
    j1, j2 = sorted((max(0, min(j1, dims.ny - 1)), max(0, min(j2, dims.ny - 1))))
    k1, k2 = sorted((max(0, min(k1, dims.nz - 1)), max(0, min(k2, dims.nz - 1))))
    return i1, i2, j1, j2, k1, k2


def resolve_operations(deck: Deck, dims: GridDimensions) -> list[Operation]:
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
    :returns: Resolved operations in file order (by `order`).
    """
    default_box = get_default_box(dims)
    current_box = default_box
    operations: list[Operation] = []

    for record in deck.records:
        if record.keyword == BOX_KEYWORD:
            tokens = tokenize(record.body)
            if len(tokens) < 6:
                warnings.warn(
                    f"`BOX` record has {len(tokens)} token(s); expected 6 "
                    "(I1 I2 J1 J2 K1 K2). Ignoring.",
                    stacklevel=4,
                )
                continue

            try:
                # Minus 1, to move from 1-based to 0-based indexing used internally
                i1, i2, j1, j2, k1, k2 = (int(t) - 1 for t in tokens[:6])
            except ValueError:
                warnings.warn(
                    f"`BOX` record has non-integer tokens {tokens[:6]!r}; ignoring.",
                    stacklevel=4,
                )
                continue
            current_box = clamp_box(i1, i2, j1, j2, k1, k2, dims)

        elif record.keyword == ENDBOX_KEYWORD:
            current_box = default_box

        elif record.keyword in OPERATOR_KEYWORDS:
            for line_index, op in enumerate(
                parse_operator_records(record.body, record.keyword, current_box, dims)
            ):
                operations.append(op._replace(order=(record.start, line_index)))

    return operations


def parse_operator_records(
    body: str,
    op: str,
    box: tuple[int, int, int, int, int, int],
    dims: GridDimensions,
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
    :param box: Ambient box active at this point in the deck.
    :param dims: Grid dimensions, used to clamp per-record box overrides.
    :yields: One `Operation` per record line (`order` field is
        a placeholder `(0, 0)` — the caller replaces it).
    """
    for line in body.split("/"):
        tokens = tokenize(line)
        if not tokens:
            continue

        record_box = box

        if op == "COPY":
            if len(tokens) < 2:
                warnings.warn(
                    f"`COPY` record {tokens!r} needs a source and target keyword; skipping.",
                    stacklevel=6,
                )
                continue
            source, target = tokens[0].upper(), tokens[1].upper()
            value = None
            extra = tokens[2:]
        else:
            if len(tokens) < 2:
                warnings.warn(
                    f"{op} record {tokens!r} needs a target keyword and value; skipping.",
                    stacklevel=6,
                )
                continue
            target = tokens[0].upper()
            source = None
            try:
                value = float(tokens[1])
            except ValueError:
                warnings.warn(
                    f"{op} record for {target!r} has non-numeric value {tokens[1]!r}; skipping.",
                    stacklevel=6,
                )
                continue
            extra = tokens[2:]

        if len(extra) >= 6:
            try:
                # Minus 1, to move from 1-based to 0-based indexing used internally
                ri1, ri2, rj1, rj2, rk1, rk2 = (int(t) - 1 for t in extra[:6])
                record_box = clamp_box(ri1, ri2, rj1, rj2, rk1, rk2, dims)
            except ValueError:
                pass  # fall back to ambient box

        yield Operation(
            op=op,
            target=target,
            value=value,
            source=source,
            box=record_box,
            order=(0, 0),
        )


def get_box_indices(
    box: tuple[int, int, int, int, int, int], dims: GridDimensions
) -> IntArray[OneDimension]:
    """
    Return the flat cell indices covered by *box*, vectorized.

    Replaces a Python-level triple-nested loop calling `flat_index` once
    per cell - which dominates runtime for large grids, since a full-grid
    (un-boxed) operator is the common case, not the exception. Order of
    the returned indices doesn't matter: every caller either broadcasts a
    scalar across them (`EQUALS`/`ADD`/`MULTIPLY`/`MAXVALUE`/`MINVALUE`) or
    uses the *same* index array on both sides of a copy
    (`array[indices] = source_array[indices]`), so any consistent
    flattening order is correct.
    """
    i1, i2, j1, j2, k1, k2 = box
    if (i1, i2, j1, j2, k1, k2) == (0, dims.nx - 1, 0, dims.ny - 1, 0, dims.nz - 1):
        return np.arange(dims.n_cells, dtype=np.intp)

    i = np.arange(i1, i2 + 1, dtype=np.intp)
    j = np.arange(j1, j2 + 1, dtype=np.intp)
    k = np.arange(k1, k2 + 1, dtype=np.intp)
    flat = i[:, None, None] + j[None, :, None] * dims.nx + k[None, None, :] * dims.nx * dims.ny
    return flat.ravel()


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
        return `None` if the source keyword has no data yet.
    """
    indices = get_box_indices(operation.box, dims)

    if operation.op == "EQUALS":
        array[indices] = operation.value
    elif operation.op == "ADD":
        array[indices] += operation.value  # type: ignore
    elif operation.op == "MULTIPLY":
        array[indices] *= operation.value  # type: ignore
    elif operation.op == "MAXVALUE":
        array[indices] = np.maximum(array[indices], operation.value)  # type: ignore
    elif operation.op == "MINVALUE":
        array[indices] = np.minimum(array[indices], operation.value)  # type: ignore
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
        array[indices] = source_array[indices]
