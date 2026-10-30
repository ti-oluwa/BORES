"""Eclipse / GRDECL deck core: scanning, tokenisation, and grid dimensions."""

import re
import typing
import warnings
from pathlib import Path

import numpy as np

from bores.errors import ValidationError
from bores.types import FloatArray, OneDimension, TextOrPath

__all__ = [
    "Deck",
    "DeckParseError",
    "NumericRun",
    "Record",
    "find_terminator",
    "first_segment",
    "parse_floats",
    "parse_numeric_runs",
    "parse_repeat_token",
    "report",
    "split_segments",
    "strip_record_notes",
    "tokenize",
]


class DeckParseError(ValidationError):
    """Raised when an Eclipse deck or one of its keyword records is malformed."""


def report(message: str, *, strict: bool, stacklevel: int = 2) -> None:
    """
    Report a problem with the deck according to the parsing mode.

    :param message: What is wrong, ending with a full stop. It should say what the parser did about
        it (or would have done), because that is what a lenient parse falls back to.
    :param strict: When `True` the problem is an error and `DeckParseError` is raised. When
        `False` it is emitted as a warning and parsing carries on.
    :param stacklevel: Stack level for the warning, relative to the caller of this function.
    :raises DeckParseError: If `strict` is `True`.
    """
    if strict:
        raise DeckParseError(
            f"{message} This is an error because the deck is being parsed with `strict=True`."
        )
    warnings.warn(message, stacklevel=stacklevel + 1)


COMMENT_RE = re.compile(r"--[^\n]*")
QUOTED_OR_COMMENT_RE = re.compile(r"""('[^'\n]*'|"[^"\n]*")|--[^\n]*""")
"""
Matches a quoted string (kept, via group 1) or a `--` comment (dropped). Scanning left to right, a
`--` that sits inside a quoted string is consumed by the string first, so it is never mistaken for
the start of a comment.
"""
NON_SPACE_RE = re.compile(r"\S")
NOTE_RE = re.compile(r"[ \t\r]*[^\s/]")
"""Matches the start of a note: text (not another `/`) following a closing `/` on its line."""
QUOTED_OR_NOTE_RE = re.compile(r"""('[^'\n]*'|"[^"\n]*")|(/)[ \t\r]*[^\s/][^\n]*""")
"""
Matches a quoted string (kept, via group 1), or a `/` followed by a note up to the end of its
line (the `/` is kept, via group 2, and the note dropped).
"""
SLASH_OR_QUOTED_RE = re.compile(r"""'[^'\n]*'|"[^"\n]*"|/""")
INCLUDE_RE = re.compile(
    r"""\bINCLUDE\b\s*
        (?:
            ['"]([^'"]+)['"]   # quoted filename
            |
            ([^\s/]+)          # unquoted filename
        )
        \s*/
        [^\n]*             # anything after the closing slash on its line is a note
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def strip_comments(text: str) -> str:
    """
    Remove `--` line comments from Eclipse text.

    A `--` inside a quoted string (e.g. `'CASE--A'`) is part of the string, not a comment. A quote
    only protects text up to the end of its own line, so an unclosed quote cannot swallow a comment.
    Newlines are always kept, so line structure is unchanged.
    """
    if "--" not in text:
        return text
    if "'" not in text and '"' not in text:
        return COMMENT_RE.sub("", text)
    return QUOTED_OR_COMMENT_RE.sub(r"\1", text)


RECORD_LIKE_RE = re.compile(r"""['"+\-.\d]""")


def looks_like_record(note: str) -> bool:
    """
    Whether a note after a record's closing `/` looks like a second record written on the same
    line, that is, begins like data and has a `/` of its own (`2 2 2 /`, `'B' 'G' 2 2 /`).

    Eclipse ignores such a record along with any other note, so this is only used to say so.
    """
    note = note.strip()
    return bool(note) and RECORD_LIKE_RE.match(note) is not None and find_terminator(note) != -1


def strip_record_notes(body: str, *, records_lost: list[str] | None = None) -> str:
    """
    Drop the note after the closing `/` of each record in a keyword body.

    Eclipse ignores whatever follows a record's closing `/` on the same line, and decks use that to
    annotate records:

        MULTIPLY
         'PERMZ' 0.2  1 46 1 112 1 1 /  Garn 3
         'PERMZ' 0.04 1 46 1 112 2 2 /  Garn 2
        /

    Without this, `Garn 3` would be read as the start of the next record. Only text is a note: a
    second `/` (as in `/ /`, or the lone `/` that closes a keyword) is not, and is kept. A `/`
    inside a quoted string is not a record end.

    :param body: Keyword body, as found between two keywords.
    :param records_lost: If given, each dropped note that `looks_like_record` is appended to it.
    :returns: `body` without the notes (`body` itself if it has none).
    """
    if "/" not in body:
        return body
    if records_lost is None:
        return QUOTED_OR_NOTE_RE.sub(r"\1\2", body)

    def drop(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            return match.group(1)
        note = match.group(0)[1:]
        if looks_like_record(note):
            records_lost.append(note.strip())
        return "/"

    return QUOTED_OR_NOTE_RE.sub(drop, body)


def find_terminator(text: str, start: int = 0, end: int | None = None) -> int:
    """
    Find the first `/` in `text[start:end]` that is not inside a quoted string.

    :param text: Text to search.
    :param start: Offset to start searching at.
    :param end: Offset to stop searching at (exclusive). Defaults to the end of `text`.
    :returns: Offset of the `/` in `text`, or `-1` if there is none.
    """
    if end is None:
        end = len(text)
    if text.find("'", start, end) == -1 and text.find('"', start, end) == -1:
        return text.find("/", start, end)
    for match in SLASH_OR_QUOTED_RE.finditer(text, start, end):
        if match.group() == "/":
            return match.start()
    return -1


def first_segment(body: str) -> str:
    """
    Return `body` up to its first `/` that is not inside a quoted string (all of `body` if there
    is none). This is the first `/`-terminated record of a keyword body, without the `/`.
    """
    terminator = find_terminator(body)
    return body if terminator == -1 else body[:terminator]


def split_segments(body: str) -> list[str]:
    """
    Split `body` on every `/` that is not inside a quoted string.

    Behaves like `body.split("/")`, except that a quoted value such as `'run/CASE1'` stays whole.
    """
    if "'" not in body and '"' not in body:
        return body.split("/")

    segments: list[str] = []
    segment_start = 0
    for match in SLASH_OR_QUOTED_RE.finditer(body):
        if match.group() == "/":
            segments.append(body[segment_start : match.start()])
            segment_start = match.end()
    segments.append(body[segment_start:])
    return segments


MAX_INCLUDE_DEPTH = 64
"""How deeply `INCLUDE` directives may nest before the deck is rejected."""

INCLUDE_SCAN_CHUNK = 1 << 24


def may_contain_include(text: str) -> bool:
    """
    Cheaply tell whether `text` can contain an `INCLUDE` directive, in any letter case.

    A case-insensitive regular expression over a large deck takes seconds even when there is
    nothing to find, so this scans in bounded chunks with plain string search instead.
    """
    length = len(text)
    if length <= INCLUDE_SCAN_CHUNK:
        return "include" in text.lower()
    for offset in range(0, length, INCLUDE_SCAN_CHUNK):
        # The extra characters stop an `INCLUDE` that straddles two chunks from being missed.
        if "include" in text[offset : offset + INCLUDE_SCAN_CHUNK + 7].lower():
            return True
    return False


def canonical_path(path: Path) -> Path:
    """Return the absolute, symlink-free form of `path`, for comparing files."""
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def resolve_includes(
    text: str,
    source_dir: Path | None,
    *,
    encoding: str = "ascii",
    strict: bool = False,
    include_stack: tuple[Path, ...] = (),
) -> str:
    """
    Recursively inline `INCLUDE 'path' /` directives.

    :param text: Eclipse text that may contain `INCLUDE` directives.
    :param source_dir: Directory the current text was read from, used to
        resolve relative include paths. `None` disables resolution (raw
        text input with no filesystem anchor); directives are dropped with
        a warning in that case (an error when `strict`).
    :param encoding: Character encoding of included files.
    :param strict: Treat an `INCLUDE` that cannot be followed (raw text input) as an error
        instead of a warning.
    :param include_stack: Files currently being expanded, outermost first. Used to detect a file
        that includes itself, directly or through other files.
    :returns: Text with every `INCLUDE` block replaced by the referenced
        file's (recursively resolved) contents, with comments removed.
    :raises DeckParseError: If an included file cannot be found, read, or decoded with `encoding`,
        if `INCLUDE` directives form a cycle, or if they nest deeper than `MAX_INCLUDE_DEPTH`.
    """
    # A commented-out `INCLUDE` must not be followed, so comments go first.
    text = strip_comments(text)
    if not may_contain_include(text):
        return text

    def replace(match: re.Match[str]) -> str:
        relative_path = (match.group(1) or match.group(2)).strip()
        if source_dir is None:
            report(
                f"`INCLUDE` directive for {relative_path!r} encountered in raw-text "
                "input with no source directory; ignoring it. Load from a file path "
                "to enable `INCLUDE` resolution.",
                strict=strict,
                stacklevel=6,
            )
            return ""

        include_path = source_dir / relative_path
        if not include_path.is_file():
            raise DeckParseError(f"`INCLUDE` references {include_path!r}, which does not exist.")

        canonical = canonical_path(include_path)
        if canonical in include_stack:
            chain = " -> ".join(str(path) for path in (*include_stack, canonical))
            raise DeckParseError(f"`INCLUDE` cycle detected: {chain}.")
        if len(include_stack) >= MAX_INCLUDE_DEPTH:
            raise DeckParseError(
                f"`INCLUDE` directives nest deeper than {MAX_INCLUDE_DEPTH} levels, "
                f"at {str(include_path)!r}."
            )

        included_text = read_deck_file(include_path, encoding=encoding)
        return resolve_includes(
            included_text,
            include_path.parent,
            encoding=encoding,
            strict=strict,
            include_stack=(*include_stack, canonical),
        )

    return INCLUDE_RE.sub(replace, text)


class NumericRun(typing.NamedTuple):
    """
    A run of consecutive entries of a numeric record, as read by `parse_numeric_runs`.

    A deck writes many equal values compactly as `N*value`, so a large uniform array is a handful
    of runs rather than millions of tokens.
    """

    entry_count: int
    """How many array entries the run stands for."""

    values: FloatArray[OneDimension] | float | None
    """
    The entries: an array of `count` values written out one by one, a single `float` for an
    `N*value` group (every entry is that value), or `None` for a bare `N*` group (`count`
    defaulted entries, which have no value of their own).
    """


REPEAT_LOOKBACK = 32
"""How far back from a `*` to look for the start of its token. Longer tokens are malformed."""

NONSPACE_RE = re.compile(r"\S*")
TRAILING_NONSPACE_RE = re.compile(r"\S*\Z")


def parse_floats(text: str, name: str) -> FloatArray[OneDimension]:
    """
    Read whitespace-separated numbers into a `float64` array.

    This is the fast path for the bulk of a large array: the text is parsed in one pass, with no
    list of token strings built along the way.

    :param text: Numbers separated by any whitespace, with no `N*` repeat groups.
    :param name: Keyword name, for error messages.
    :returns: The numbers in order (empty for empty or blank text).
    :raises DeckParseError: If anything in `text` is not a number.
    """
    if not text or text.isspace():
        # `numpy.fromstring` reads blank text as `[-1.0]`, so it must not be asked to.
        return np.empty(0, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            # Older numpy only warns about, and stops at, text it cannot read.
            warnings.simplefilter("error", DeprecationWarning)
            return typing.cast(
                "FloatArray[OneDimension]", np.fromstring(text, dtype=np.float64, sep=" ")
            )
    except (ValueError, DeprecationWarning, UnicodeEncodeError):
        pass

    # Something is not a number. Find it so the message can name it.
    for token in text.split():
        try:
            float(token)
        except ValueError:
            raise DeckParseError(f"{name} contains a non-numeric value {token!r}.") from None
    raise DeckParseError(f"{name} contains a value that could not be read as a number.")


def parse_numeric_runs(text: str, *, name: str) -> list[NumericRun]:
    """
    Read the numbers of a record body, keeping `N*value` and `N*` groups as single runs.

    :param text: Record body up to its terminating `/`.
    :param name: Keyword name, for error messages.
    :returns: The runs, in order. Groups of zero entries are left out.
    :raises DeckParseError: If anything is not a number or a well-formed repeat group.
    """
    runs: list[NumericRun] = []
    if "*" not in text:
        values = parse_floats(text, name)
        if values.size:
            runs.append(NumericRun(int(values.size), values))
        return runs

    position = 0
    star = text.find("*")
    while star != -1:
        window_start = max(position, star - REPEAT_LOOKBACK)
        trailing = TRAILING_NONSPACE_RE.search(text[window_start:star])
        assert trailing is not None  # `\S*\Z` always matches, if only the empty string
        start = window_start + trailing.start()
        end = NONSPACE_RE.match(text, star + 1).end()  # type: ignore[union-attr]
        digits = text[start:star]
        too_long = (
            start == window_start
            and window_start > position
            and not text[window_start - 1].isspace()
        )
        if too_long or not (digits.isascii() and digits.isdigit()):
            raise DeckParseError(
                f"{name}: {text[start:end]!r} is not a valid repeat group "
                "(expected N*value, or N* for N defaulted entries)."
            )

        if start > position:
            values = parse_floats(text[position:start], name)
            if values.size:
                runs.append(NumericRun(int(values.size), values))

        count = int(digits)
        value_text = text[star + 1 : end]
        if count > 0:
            if not value_text:
                runs.append(NumericRun(count, None))
            else:
                try:
                    runs.append(NumericRun(count, float(value_text)))
                except ValueError:
                    raise DeckParseError(
                        f"{name} contains a non-numeric value {value_text!r} "
                        f"in the repeat group {text[start:end]!r}."
                    ) from None
        position = end
        star = text.find("*", end)

    if position < len(text):
        values = parse_floats(text[position:], name)
        if values.size:
            runs.append(NumericRun(int(values.size), values))
    return runs


def decode_deck_bytes(data: bytes, *, source: str, encoding: str) -> str:
    """
    Decode raw deck bytes, turning a decoding failure into an error that says what to do.

    :param data: The deck's raw bytes.
    :param source: What `data` came from, for the error message (a file path, or `"bytes input"`).
    :param encoding: Character encoding to decode with.
    :returns: The decoded text.
    :raises DeckParseError: If `encoding` is unknown or the bytes are not valid in it.
    """
    try:
        return data.decode(encoding)
    except LookupError as exc:
        raise DeckParseError(
            f"Cannot read deck {source}: {encoding!r} is not a known text encoding."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DeckParseError(
            f"Cannot read deck {source} as {encoding!r}: {exc.reason} at byte {exc.start}. "
            "The file is probably in a different encoding. Check the file's encoding and pass it "
            "as `encoding=`, e.g. `'utf-8'` (the usual choice when the file has non-ASCII "
            "characters in comments or names), `'utf-8-sig'` (when it starts with a byte order "
            "mark) or `'latin-1'` (reads any file, but may show accented text wrongly)."
        ) from exc


def read_deck_file(path: Path, *, encoding: str) -> str:
    """
    Read a deck file's text.

    :param path: The file to read.
    :param encoding: Character encoding to read it with.
    :returns: The file's text.
    :raises DeckParseError: If the file cannot be read, or is not valid in `encoding`.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DeckParseError(f"Cannot read deck file {path!r}: {exc}") from exc
    return decode_deck_bytes(data, source=f"file {str(path)!r}", encoding=encoding)


def is_lone_keyword(source: str) -> bool:
    """Whether a single-line `str` is nothing but one keyword, such as `METRIC`."""
    return KEYWORD_LINE_RE.fullmatch(source) is not None


def resolve_source(source: TextOrPath, *, encoding: str, strict: bool = False) -> str:
    """
    Coerce `source` (path, raw text, or bytes) to a single fully
    `INCLUDE`-resolved, comment-stripped text blob.

    :param source: Path, raw deck text string, or raw bytes. A `str` that spans more than one
        line is raw deck text. A single-line `str` is a file path if a file exists there, and
        otherwise raw deck text if it is a lone keyword (such as `METRIC`). Anything else is
        taken to be a path to a file that does not exist, which is an error.
    :param encoding: Character encoding for file/bytes input.
    :param strict: Treat problems that a lenient parse only warns about as errors.
    :returns: Clean text ready for `Deck` scanning.
    :raises DeckParseError: If a file cannot be read.
    """
    source_dir: Path | None = None
    include_stack: tuple[Path, ...] = ()

    if isinstance(source, bytes):
        text = decode_deck_bytes(source, source="bytes input", encoding=encoding)
    elif isinstance(source, Path):
        source_dir = source.parent
        include_stack = (canonical_path(source),)
        text = read_deck_file(source, encoding=encoding)
    elif "\n" in source or "\r" in source:  # type: ignore[operator]
        text = typing.cast(str, source)
    else:
        candidate = Path(source)  # type: ignore[arg-type]
        try:
            is_file = candidate.is_file()
        except (OSError, ValueError):
            is_file = False

        if is_file:
            source_dir = candidate.parent
            include_stack = (canonical_path(candidate),)
            text = read_deck_file(candidate, encoding=encoding)
        elif is_lone_keyword(source):  # type: ignore[arg-type]
            text = typing.cast(str, source)
        else:
            raise DeckParseError(
                f"Cannot read deck file. No file exists at {source[:80]!r}. A single-line `str` "
                "that does not name an existing file is read as raw deck text only when it is a "
                "lone keyword such as `METRIC`. Use `DeckFile.from_text` for raw text."
            )

    return resolve_includes(
        text,
        source_dir,
        encoding=encoding,
        strict=strict,
        include_stack=include_stack,
    )


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
        elements long from a handful of `N*value` groups (`ArrayKeyword`),
        where expanding to `N` separate tokens, and then parsing the same
        numeric string `N` redundant times, is unnecessary overhead.
    :returns: Flat list of tokens (quotes stripped; repeats expanded or not
        per `expand_repeats`).
    """
    placeholders: list[str] = []

    def stash(match: re.Match[str]) -> str:
        placeholders.append(match.group(2))
        return f"\x00{len(placeholders) - 1}\x00"

    has_quotes = "'" in text or '"' in text
    stashed_text = QUOTED_RE.sub(stash, text) if has_quotes else text

    tokens: list[str] = []
    for raw_token in stashed_text.split():
        # Most tokens in a large array are plain numbers; skip the regex for them.
        repeat = parse_repeat_token(raw_token) if "*" in raw_token else None
        if repeat is None:
            tokens.append(raw_token)
            continue

        count, value = repeat
        if expand_repeats:
            tokens.extend([value] * count)
        else:
            tokens.append(raw_token)

    if not has_quotes:
        return tokens

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
    Raw text after the keyword (untokenised): up to, and not including, its terminating `/`
    when that is the only `/` of the keyword, which is the case for every single-record keyword
    such as `PORO` or `SPECGRID`. A keyword with several `/`-terminated records (a table, or an
    operator block) keeps all of them, `/` included, up to the next keyword. Empty for a
    bare/nullary keyword that takes no data section.
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

    line: int = 0
    """
    One-based line of the keyword name in the deck text. When the deck pulls in other files with
    `INCLUDE`, this counts lines in the combined text, with each included file placed where its
    `INCLUDE` was. `0` when unknown.
    """


# A keyword name is a bare token of letters, digits and underscores starting
# with a letter, 1-8 characters (Eclipse's own limit), optionally ending in `-`
# (`MULTX-`, `MULTY-`, `MULTZ-`), standing alone on its own line (only
# whitespace or end-of-line after it). Eclipse keywords are case-insensitive, so
# `poro` is the same keyword as `PORO`; `Deck` upper-cases every name it records.
# The underscore is intentional because some simulator extensions use keywords
# like `COORD_V`.
KEYWORD_LINE_RE = re.compile(
    r"^[ \t]*(?P<keyword>[A-Z][A-Z0-9_]{0,7}-?)[ \t]*(?:[=*#~_+:.|<>^!%&@]+[ \t]*)?\r?$",
    re.MULTILINE | re.IGNORECASE,
)

# Keywords whose data is free text with no `/` terminator, i.e, the data simply
# runs until the next keyword line. Without this exception, `_scan`'s
# general "no slash before the next keyword line -> bare/nullary" rule would
# (incorrectly) swallow their body as empty, since they never contain a
# literal `/` in practice.
FREE_TEXT_KEYWORDS = frozenset({"TITLE"})


def has_unterminated_quote(text: str) -> bool:
    """
    Whether `text` has a `'` or `"` that is not closed on its own line.

    Quoted strings never span lines in an Eclipse deck, so a quote character left over once every
    complete quoted string is removed was opened and never closed.
    """
    if "'" not in text and '"' not in text:
        return False
    remainder = QUOTED_RE.sub("", text)
    return "'" in remainder or '"' in remainder


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
    unless another keyword line is encountered first, in which case the
    first keyword is treated as bare/nullary (e.g. `ENDBOX`, `ECHO`,
    `RUNSPEC`) with an empty body, and the second keyword line starts a
    new record. This mirrors Eclipse's own grammar without needing a
    hardcoded list of nullary keywords.

    A `/` inside a quoted string (e.g. `'run/CASE1'`) is part of the string and does not end a
    record.

    Note:
        A keyword that has data but no terminating `/` before the next keyword (or the end of
        the deck) cannot be read: its data is dropped and the keyword becomes nullary. Well-formed
        Eclipse decks always terminate data blocks with `/`, so this only arises with truncated or
        hand-written files. It is reported with a warning, or raised as a `DeckParseError` when
        `strict`.
    """

    __slots__ = ("_keyword_records", "records", "strict")

    def __init__(self, text: str, *, strict: bool = False) -> None:
        """
        Initialize the deck.

        :param text: Fully comment-stripped, `INCLUDE`-resolved Eclipse
            text. It is scanned once and not kept: every record holds its own body, so the
            deck's memory is the record bodies rather than those plus a second copy of the text.
        :param strict: Raise a `DeckParseError` for malformed input that a lenient parse only
            warns about (here: data without a terminating `/`, unterminated quotes). The keyword
            parsers read this too, so one flag sets the mode for the whole deck.
        :raises DeckParseError: If `strict` and the deck is malformed.
        """
        self.strict = strict
        self.records, self._keyword_records = self._scan(text, strict)

    @staticmethod
    def _scan(text: str, strict: bool = False) -> ScanResult:
        """
        Scan `text` into an ordered list of `Record` objects together with an
        index mapping upper-cased keywords to all matching records.

        :param text: Clean Eclipse text (no comments, no `INCLUDE` directives).
        :param strict: Raise instead of warning about malformed records.
        :returns: `ScanResult` containing records in file order and a keyword index.
        """
        keyword_lines: list[re.Match[str]] = []
        for match in KEYWORD_LINE_RE.finditer(text):
            # The line right after a free-text keyword is that keyword's text
            # (e.g. a title such as `SPE1`), even when it looks like a keyword.
            if (
                keyword_lines
                and keyword_lines[-1].group("keyword").upper() in FREE_TEXT_KEYWORDS
                and not text[keyword_lines[-1].end() : match.start()].strip()
            ):
                continue
            keyword_lines.append(match)

        records: list[Record] = []
        keyword_records: dict[str, list[Record]] = {}

        counted_to = 0
        line = 1
        for idx, match in enumerate(keyword_lines):
            keyword = match.group("keyword").upper()
            body_start = match.end()
            line += text.count("\n", counted_to, match.start())
            counted_to = match.start()

            # The next keyword line (if any) bounds how far this keyword's body
            # may extend, even if no '/' is found before it.
            next_keyword_start = (
                keyword_lines[idx + 1].start() if idx + 1 < len(keyword_lines) else len(text)
            )
            terminator = find_terminator(text, body_start, next_keyword_start)

            if terminator == -1:
                if keyword in FREE_TEXT_KEYWORDS:
                    # No `/` at all, and none expected - the free text
                    # itself runs until the next keyword line.
                    record = Record(
                        keyword=keyword,
                        body=text[body_start:next_keyword_start],
                        start=match.start(),
                        line=line,
                        end=next_keyword_start,
                    )
                else:
                    # No terminator before the next keyword line (or end of
                    # file): treat as bare/nullary keyword. That is only right when there is
                    # nothing between the two; data here has been left unterminated.
                    if (
                        next_keyword_start > body_start
                        and not text[body_start:next_keyword_start].isspace()
                    ):
                        next_keyword = (
                            keyword_lines[idx + 1].group("keyword").upper()
                            if idx + 1 < len(keyword_lines)
                            else None
                        )
                        before = (
                            f"before the next keyword `{next_keyword}`"
                            if next_keyword is not None
                            else "before the end of the deck"
                        )
                        preview = " ".join(text[body_start : body_start + 200].split())[:60]
                        report(
                            f"`{keyword}` has data but no terminating '/' {before}. The data "
                            f"is ignored and `{keyword}` is treated as having no data "
                            f"(data starts with {preview!r}).",
                            strict=strict,
                            stacklevel=4,
                        )
                    record = Record(
                        keyword=keyword,
                        body="",
                        start=match.start(),
                        line=line,
                        end=body_start,
                    )
            else:
                # Capture the entire span up to the next keyword line, not just
                # up to the first `/`. Single-record keywords (SPECGRID, PORO, ...)
                # only ever have one `/` in this span, so this changes nothing for
                # them. Multi-row keywords (PVTO, PVTG, FAULTS, DENSITY with several
                # regions, multi-entry DATES, ...) legitimately contain several
                # `/`-terminated rows before the next keyword line, and their own
                # `Keyword.parse()` already splits on `/` internally to walk them -
                # truncating here silently dropped every row after the first.
                #
                # When nothing but whitespace follows that one `/` (every single-record
                # keyword, which is where the large arrays are), the body stops before it.
                # Parsers then take the body as is, instead of each cutting a second copy of a
                # possibly huge array out of it.
                #
                # A note after that `/` on its own line (`... / Garn 3`) is not data, so it
                # does not make the keyword a multi-record one either.
                free_text = keyword in FREE_TEXT_KEYWORDS
                tail_start = terminator + 1
                records_lost: list[str] = []
                if not free_text:
                    line_end = text.find("\n", tail_start, next_keyword_start)
                    if line_end == -1:
                        line_end = next_keyword_start
                    if NOTE_RE.match(text, tail_start, line_end):
                        if looks_like_record(text[tail_start:line_end]):
                            records_lost.append(text[tail_start:line_end].strip())
                        tail_start = line_end

                only_record = (
                    not free_text
                    and NON_SPACE_RE.search(text, tail_start, next_keyword_start) is None
                )
                body = text[body_start : terminator if only_record else next_keyword_start]
                if not (free_text or only_record):
                    # The first note was looked at above, so only the later ones are collected.
                    later_notes: list[str] = []
                    body = strip_record_notes(body, records_lost=later_notes)
                    records_lost.extend(later_notes[1:] if records_lost else later_notes)
                for note in records_lost[:1]:
                    report(
                        f"`{keyword}`: text after a record's closing '/' is ignored, as Eclipse "
                        f"does, but {note!r} looks like another record on the same line. Write "
                        "each record on its own line.",
                        strict=strict,
                        stacklevel=4,
                    )
                record = Record(
                    keyword=keyword,
                    body=body,
                    start=match.start(),
                    line=line,
                    end=next_keyword_start,
                )

            if keyword not in FREE_TEXT_KEYWORDS and has_unterminated_quote(record.body):
                report(
                    f"`{keyword}` contains a quote that is never closed on its line, so the "
                    "quoted value is read incorrectly "
                    f"(near {' '.join(record.body.split())[:60]!r}).",
                    strict=strict,
                    stacklevel=4,
                )

            records.append(record)
            keyword_records.setdefault(keyword, []).append(record)
        return ScanResult(records=records, keyword_records=keyword_records)

    def describe_location(self, keyword: str, *, limit: int = 5) -> str:
        """
        Say where `keyword` is in the deck, for use in error messages.

        :param keyword: Keyword name.
        :param limit: Most line numbers to list before summarising the rest.
        :returns: `"line 155"` for one occurrence, `"lines 155, 340"` for several, or an empty
            string if the keyword is absent or its lines are unknown.
        """
        lines = [record.line for record in self.get_records_for(keyword) if record.line > 0]
        if not lines:
            return ""
        if len(lines) == 1:
            return f"line {lines[0]}"
        shown = ", ".join(str(line) for line in lines[:limit])
        extra = f" and {len(lines) - limit} more" if len(lines) > limit else ""
        return f"lines {shown}{extra}"

    def get_records_for(self, keyword: str) -> list[Record]:
        """Return every record for `keyword`, in file order."""
        return self._keyword_records.get(keyword.upper(), [])

    def get_first_record_for(self, keyword: str) -> Record | None:
        """Return the first record for `keyword`, or `None` if absent."""
        records = self._keyword_records.get(keyword.upper())
        return records[0] if records else None

    def get_single_record_for(self, keyword: str) -> Record | None:
        """
        Return the record for a keyword that may only appear once, or `None` if absent.

        Only the first occurrence is ever read, so a repeat is reported (a warning, or a
        `DeckParseError` when `strict`) instead of its data being silently ignored.
        """
        records = self._keyword_records.get(keyword.upper())
        if not records:
            return None
        if len(records) > 1:
            report(
                f"`{keyword.upper()}` appears {len(records)} times in the deck but may only "
                "appear once. Only the first occurrence is read.",
                strict=self.strict,
                stacklevel=3,
            )
        return records[0]

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
