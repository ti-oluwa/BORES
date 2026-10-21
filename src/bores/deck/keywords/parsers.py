"""
Reusable value parsers for `bores.deck.keywords.base.Field.type`.

A keyword's job is to read a deck, not to judge it, so most fields use a plain
`int` or `float`. The parsers here cover the few cases where a plain conversion
is not enough: values that Eclipse accepts in several spellings, or values that
only make sense inside a fixed range.
"""

import typing

import numpy as np

from bores.errors import ValidationError

__all__ = [
    "FRACTION_TOLERANCE",
    "fault_face",
    "fraction",
    "preferred_phase",
    "uppercase",
    "yes_no",
]

FRACTION_TOLERANCE = 1e-6
"""
How far outside `[0, 1]` a fraction may land before it is rejected.

Decks exported from other tools often carry round-off such as `1.0000001` for
an end-point saturation, which is not an input error.
"""


def uppercase(value: typing.Any) -> str:
    """
    Convert a raw token to upper case, dropping padding spaces from quoted values.

    Eclipse keyword options are case-insensitive, so every option-valued field
    runs its token through this before it is checked against the field's options.

    :param value: Raw token.
    :returns: The token in upper case.
    """
    return str(value).strip().upper()


def yes_no(value: typing.Any) -> bool:
    """
    Convert an Eclipse `YES` / `NO` token to a `bool`.

    :param value: Raw token, or an already converted `bool`.
    :returns: `True` for `YES`, `False` for `NO`.
    :raises ValidationError: If the token is neither `YES` nor `NO`.
    """
    if isinstance(value, bool):
        return value

    token = uppercase(value)
    if token == "YES":
        return True
    if token == "NO":
        return False
    raise ValidationError(f"Expected 'YES' or 'NO', got {value!r}.")


def fraction(value: typing.Any) -> np.float64:
    """
    Convert a raw token to a float that must lie in `[0, 1]`.

    Used by saturation and relative permeability columns, where a value outside
    that range can only be a typo or a column in the wrong place.

    :param value: Raw token.
    :returns: The value as a `np.float64`.
    :raises ValidationError: If the value is outside `[0, 1]` by more than
        `FRACTION_TOLERANCE`.
    """
    number = np.float64(value)
    if not -FRACTION_TOLERANCE <= number <= 1.0 + FRACTION_TOLERANCE:
        raise ValidationError(f"Expected a value between 0 and 1, got {value!r}.")
    return number


def fault_face(value: typing.Any) -> str:
    """
    Convert a fault face token to the `X`, `X-`, `Y`, `Y-`, `Z`, `Z-` spelling.

    Eclipse accepts `I`, `J` and `K` as synonyms for `X`, `Y` and `Z`, and a
    trailing `+` on the positive faces.

    :param value: Raw token.
    :returns: The face in `X`/`Y`/`Z` spelling, with `-` kept for negative faces.
    """
    face = uppercase(value).removesuffix("+")
    axis, negative = face.removesuffix("-"), face.endswith("-")
    axis = {"I": "X", "J": "Y", "K": "Z"}.get(axis, axis)
    return f"{axis}-" if negative else axis


def preferred_phase(value: typing.Any) -> str:
    """
    Convert a well preferred phase token to `OIL`, `WATER`, `GAS` or `LIQ`.

    Eclipse accepts `WAT` as a short form of `WATER`.

    :param value: Raw token.
    :returns: The phase name, with `WAT` expanded to `WATER`.
    """
    phase = uppercase(value)
    return "WATER" if phase == "WAT" else phase
