"""Eclipse / GRDECL deck core: scanning, tokenisation, and grid dimensions."""

import re
import typing
import warnings
from pathlib import Path

from bores.errors import ValidationError
from bores.types import TextOrPath

__all__ = [
    "Deck",
    "DeckParseError",
    "Record",
    "parse_repeat_token",
    "tokenize",
]


class DeckParseError(ValidationError):
    """Raised when an Eclipse deck or one of its keyword records is malformed."""


COMMENT_RE = re.compile(r"--[^\n]*")
INCLUDE_RE = re.compile(
    r"""\bINCLUDE\b\s*
        (?:
            ['"]([^'"]+)['"]   # quoted filename
            |
            ([^\s/]+)          # unquoted filename
        )
        \s*/
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def strip_comments(text: str) -> str:
    """Remove `--` line comments from Eclipse text."""
    return COMMENT_RE.sub("", text)


def resolve_includes(text: str, source_dir: Path | None) -> str:
    """
    Recursively inline `INCLUDE 'path' /` directives.

    :param text: Eclipse text that may contain `INCLUDE` directives.
    :param source_dir: Directory the current text was read from, used to
        resolve relative include paths. `None` disables resolution (raw
        text input with no filesystem anchor); directives are dropped with
        a warning in that case.
    :returns: Text with every `INCLUDE` block replaced by the referenced
        file's (recursively resolved) contents.
    :raises DeckParseError: If an included file cannot be found or read.
    """

    def replace(match: re.Match[str]) -> str:
        relative_path = (match.group(1) or match.group(2)).strip()
        if source_dir is None:
            warnings.warn(
                f"`INCLUDE` directive for {relative_path!r} encountered in raw-text "
                "input with no source directory; ignoring. Load from a file path "
                "to enable `INCLUDE` resolution.",
                stacklevel=6,
            )
            return ""

        include_path = source_dir / relative_path
        if not include_path.is_file():
            raise DeckParseError(f"`INCLUDE` references {include_path!r}, which does not exist.")
        try:
            included_text = include_path.read_text(encoding="ascii", errors="replace")
        except OSError as exc:
            raise DeckParseError(f"Cannot read `INCLUDE` file {include_path!r}: {exc}") from exc
        return resolve_includes(included_text, include_path.parent)

    return INCLUDE_RE.sub(replace, text)


def resolve_source(source: TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` (path, raw text, or bytes) to a single fully
    `INCLUDE`-resolved, comment-stripped text blob.

    :param source: Path, raw deck text string, or raw bytes.
    :param encoding: Character encoding for file/bytes input.
    :returns: Clean text ready for `Deck` scanning.
    :raises DeckParseError: If a file cannot be read.
    """
    source_dir: Path | None = None

    if isinstance(source, bytes):
        text = source.decode(encoding)
    elif isinstance(source, Path):
        source_dir = source.parent
        try:
            text = source.read_text(encoding=encoding)
        except OSError as exc:
            raise DeckParseError(f"Cannot read deck file {source!r}: {exc}") from exc
    else:
        candidate = Path(source)  # type: ignore[arg-type]
        if candidate.is_file():
            source_dir = candidate.parent
            try:
                text = candidate.read_text(encoding=encoding)
            except (OSError, UnicodeDecodeError) as exc:
                raise DeckParseError(f"Cannot read deck file {source!r}: {exc}") from exc
        else:
            raise DeckParseError(f"Cannot read deck file. Invalid source: {source!r}")

    return resolve_includes(text, source_dir)


# Matches "N*value" repeat syntax. Value must be non-empty; a bare "N*" is
# treated as N repetitions of the empty string and rejected by callers that
# call float() on the result. We keep the permissive match here and let
# callers produce a meaningful error message.
REPEAT_RE = re.compile(r"^(\d+)\*(.*)$")
QUOTED_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")


def parse_repeat_token(token: str) -> tuple[int, str] | None:
    """
    If `token` is `N*value` repeat syntax, return `(N, value)`; else `None`.

    Exposed so callers that need to fill a large array without ever
    materializing `N` copies of `value` (see `tokenize`'s `expand_repeats`
    parameter) can detect and handle repeat groups themselves.
    """
    match = REPEAT_RE.match(token)
    if match is None:
        return None

    count = int(match.group(1))
    value = match.group(2)

    # Eclipse "N*" means N defaulted fields.
    if value == "":
        value = "1*"
    return count, value


def tokenize(text: str, *, expand_repeats: bool = True) -> list[str]:
    """
    Split text into whitespace-separated tokens, expanding `N*value`
    repeat syntax and preserving quoted strings (with embedded whitespace)
    as single tokens.

    Examples:

    ```md
    "100*0"        -> ["0"] * 100
    "3*1.5"        -> ["1.5", "1.5", "1.5"]
    "MY FAULT 1" -> ["MY FAULT", "1"]
    ```

    **Note**:
        A bare "N*" expands to N occurrences of the Eclipse default
        designator ("1*"). This preserves the distinction between an
        explicit default and an empty string.

        Callers that convert tokens to `float` will receive a
        `ValueError`; they should surface this as a
        `DeckParseError` with context about which keyword failed.

    :param text: Comment-stripped Eclipse text.
    :param expand_repeats: When `True` (default), `N*value` expands to `N`
        literal tokens. This is correct for small, fixed-shape records. When
        `False`, a repeat group is returned as a single unexpanded
        `"N*value"` token; use `parse_repeat_token` to inspect it. This
        exists for callers filling per-cell arrays that may be millions of
        elements long from a handful of `N*value` groups (`GridArrayKeyword`),
        where expanding to `N` separate tokens, and then parsing the same
        numeric string `N` redundant times which is just unnecessary overhead.
    :returns: Flat list of tokens (quotes stripped; repeats expanded or not
        per `expand_repeats`).
    """
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(2))
        return f"\x00{len(placeholders) - 1}\x00"

    stashed_text = QUOTED_RE.sub(stash, text)

    tokens: list[str] = []
    for raw_token in stashed_text.split():
        repeat = parse_repeat_token(raw_token)
        if repeat is None:
            tokens.append(raw_token)
            continue

        count, value = repeat
        if expand_repeats:
            tokens.extend([value] * count)
        else:
            tokens.append(raw_token)

    def unstash(token: str) -> str:
        if token.startswith("\x00") and token.endswith("\x00"):
            try:
                return placeholders[int(token[1:-1])]
            except (ValueError, IndexError):
                return token
        return token

    return [unstash(token) for token in tokens]


class Record(typing.NamedTuple):
    """One keyword occurrence in a scanned deck, in file order."""

    keyword: str
    """Upper-cased keyword name as it appeared in the deck."""

    body: str
    """
    Raw text between the keyword and its terminating `/` (untokenised),
    or empty string for a bare/nullary keyword that takes no data section.
    """
    # NOTE: Record could essentially just store the start and end position of the
    # keyword record body in the original text buffer and not store `body` at all.
    # However, that would imply for every `Keyword.parse` a slice (copy) from the body is taken.
    # This may look like memory savings through lazy evaluation of record body when needed,
    # but the cost of slicing a string is *O(length of slice)*, because strings are immutable
    # and slices copy data. This can be very expensive especially for repeated lookup of keywords
    # that can have very large record bodies e.g `PORO` or any array kind keyword. So why not just
    # pay that cost upfront (at the expense of a little more memory usage). Atleast that's why the
    # `Deck` and `DeckFile` API exists - upfront pre-parsing of Eclipse deck text records.

    start: int
    """Character offset of the keyword name in the source text."""

    end: int
    """
    Character offset just past the terminating `/` (or just past the
    keyword name itself for a bare/nullary keyword).
    """


# A keyword name is a bare uppercase token, 1-8 characters (Eclipse's own
# limit), optionally ending in `-` (`MULTX-`, `MULTY-`, `MULTZ-`),
# standing alone on its own line (only whitespace or end-of-line after it).
# The underscore in `[A-Z0-9_]` is intentional because some simulator extensions
# use keywords like `COORD_V`.
KEYWORD_LINE_RE = re.compile(
    r"^[ \t]*(?P<keyword>[A-Z][A-Z0-9_]{0,7}-?)[ \t]*\r?$",
    re.MULTILINE,
)


# Keywords whose data is free text with no `/` terminator, i.e, the data simply
# runs until the next keyword line. Without this exception, `_scan`'s
# general "no slash before the next keyword line -> bare/nullary" rule would
# (incorrectly) swallow their body as empty, since they never contain a
# literal `/` in practice.
FREE_TEXT_KEYWORDS = frozenset({"TITLE"})


class ScanResult(typing.NamedTuple):
    """Result of scanning an Eclipse deck."""

    records: list[Record]
    keyword_records: dict[str, list[Record]]


class Deck:
    """
    A single linear scan of comment-stripped, include-resolved Eclipse text
    into an ordered sequence of `Record` objects.

    Scanning is performed once on initiaization; every subsequent lookup
    re-uses the same `records` list, so the cost of finding keyword and
    operator occurrences is paid exactly once per deck regardless of how
    many keywords are subsequently requested.

    **Scanning strategy**: Eclipse keywords always start at the beginning
    of a line, alone (nothing else on that line). A keyword's data section
    is everything from there up to and including its terminating `/` -
    *unless* another keyword line is encountered first, in which case the
    first keyword is treated as bare/nullary (e.g. `ENDBOX`, `ECHO`,
    `RUNSPEC`) with an empty body, and the second keyword line starts a
    new record. This mirrors Eclipse's own grammar without needing a
    hardcoded list of nullary keywords.

    Note:
        If the **last** keyword in a deck has data but no terminating
        `/`, that data is silently dropped (the keyword becomes nullary).
        Well-formed Eclipse decks always terminate data blocks with `/`;
        this edge case only arises with truncated or hand-written files.
    """

    __slots__ = ("_hash", "_keyword_records", "records", "text")

    def __init__(self, text: str) -> None:
        """
        :param text: Fully comment-stripped, `INCLUDE`-resolved Eclipse
            text.
        """
        self.text = text
        self.records, self._keyword_records = self._scan(text)
        self._hash: int | None = None

    @staticmethod
    def _scan(text: str) -> ScanResult:
        """
        Scan `text` into an ordered list of `Record` objects together with an
        index mapping upper-cased keywords to all matching records.

        :param text: Clean Eclipse text (no comments, no `INCLUDE` directives).
        :returns: `ScanResult` containing records in file order and a keyword index.
        """
        keyword_lines = list(KEYWORD_LINE_RE.finditer(text))

        records: list[Record] = []
        keyword_records: dict[str, list[Record]] = {}

        for idx, match in enumerate(keyword_lines):
            keyword = match.group("keyword").upper()
            body_start = match.end()

            # The next keyword line (if any) bounds how far this keyword's body
            # may extend, even if no '/' is found before it.
            next_keyword_start = (
                keyword_lines[idx + 1].start() if idx + 1 < len(keyword_lines) else len(text)
            )
            window = text[body_start:next_keyword_start]
            slash_position = window.find("/")

            if slash_position == -1:
                if keyword in FREE_TEXT_KEYWORDS:
                    # No `/` at all, and none expected - the free text
                    # itself runs until the next keyword line.
                    record = Record(
                        keyword=keyword,
                        body=window,
                        start=match.start(),
                        end=next_keyword_start,
                    )
                else:
                    # No terminator before the next keyword line (or end of
                    # file): treat as bare/nullary keyword.
                    record = Record(
                        keyword=keyword,
                        body="",
                        start=match.start(),
                        end=body_start,
                    )
            else:
                # Capture the *entire* span up to the next keyword line, not just
                # up to the first `/`. Single-record keywords (SPECGRID, PORO, ...)
                # only ever have one `/` in this span, so this changes nothing for
                # them. Multi-row keywords (PVTO, PVTG, FAULTS, DENSITY with several
                # regions, multi-entry DATES, ...) legitimately contain several
                # `/`-terminated rows before the next keyword line, and their own
                # `Keyword.parse()` already splits on `/` internally to walk them -
                # truncating here silently dropped every row after the first.
                record = Record(
                    keyword=keyword,
                    body=window,
                    start=match.start(),
                    end=next_keyword_start,
                )

            records.append(record)
            keyword_records.setdefault(keyword, []).append(record)
        return ScanResult(records=records, keyword_records=keyword_records)

    def get_records_for(self, keyword: str) -> list[Record]:
        """Return every record for `keyword`, in file order."""
        return self._keyword_records.get(keyword.upper(), [])

    def get_first_record_for(self, keyword: str) -> Record | None:
        """Return the first record for `keyword`, or `None` if absent."""
        records = self._keyword_records.get(keyword.upper())
        return records[0] if records else None

    @property
    def keywords(self) -> list[str]:
        """Return every unique keyword in the deck, in file order."""
        return list(self._keyword_records.keys())

    def has(self, keyword: str) -> bool:
        """
        Return whether `keyword` has a literal record of its own in the
        deck (i.e. appears as its own keyword line).

        This does not know about `BOX`/`EQUALS`/`ADD`/`MULTIPLY`/`COPY`/
        `MAXVALUE`/`MINVALUE` operator targets. A keyword set purely via
        e.g. `EQUALS 'DZ' ... /` has no record of its own and this
        returns `False` for it, even though its value is fully resolvable.

        `Deck` is a pure scan of literal text; operator-target resolution
        needs `GridDimensions` (not yet known at this point) and lives one
        layer up, in `bores.deck.file.DeckFile.has`, which checks both.
        """
        return keyword.upper() in self._keyword_records

    def __hash__(self) -> int:
        if self._hash is None:
            # Record objects should be hashable as they are tuples of hashable fields
            self._hash = hash((self.text, tuple(self.records)))
        return self._hash
