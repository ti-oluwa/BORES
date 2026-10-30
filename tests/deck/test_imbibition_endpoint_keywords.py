"""Tests for the imbibition end-point scaling keywords and the options that switch hysteresis on."""

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.deck.file import DEFAULT_KEYWORDS

IMBIBITION_ENDPOINTS = (
    "ISWL",
    "ISWLPC",
    "ISWCR",
    "ISWU",
    "ISGL",
    "ISGLPC",
    "ISGCR",
    "ISGU",
    "ISOWCR",
    "ISOGCR",
    "IKRW",
    "IKRWR",
    "IKRO",
    "IKRORW",
    "IKRORG",
    "IKRG",
    "IKRGR",
    "IPCW",
    "IPCG",
    "SWLPC",
    "SGLPC",
)


def _deck(body: str) -> DeckFile:
    return DeckFile(
        f"METRIC\nRUNSPEC\nDIMENS\n 2 1 1 /\nSATOPTS\n HYSTER /\nENDSCALE\n/\nPROPS\n{body}"
    )


def test_every_imbibition_endpoint_keyword_is_registered():
    names = {keyword.name for keyword in DEFAULT_KEYWORDS}
    assert set(IMBIBITION_ENDPOINTS) <= names


@pytest.mark.parametrize("name", IMBIBITION_ENDPOINTS)
def test_an_imbibition_endpoint_is_read_per_cell(name):
    deck_file = _deck(f"{name}\n 0.21 0.34 /\n")
    np.testing.assert_allclose(deck_file.get(name), [0.21, 0.34])


def test_an_imbibition_endpoint_a_deck_leaves_out_is_not_invented():
    deck_file = _deck("ISGCR\n 0.1 0.2 /\n")
    assert deck_file.get("ISWL") is None


def test_hysteresis_is_switched_on_by_the_satopts_option():
    assert _deck("ISGCR\n 0.1 0.2 /\n").get("SATOPTS") == ["HYSTER"]
