"""Tests for two-phase saturation function loading from a deck."""

import numpy as np
import pytest

from bores.blackoil.satfunc.regions import SatFunc
from bores.blackoil.satfunc.relperm.tables import (
    ThreePhaseRelPermTable,
    TwoPhaseRelPermTable,
    detect_saturation_systems,
)
from bores.deck import DeckFile
from bores.errors import ValidationError

SGOF = "SGOF\n 0.0 0.0 1.0 0.0\n 0.5 0.6 0.1 0.0\n 0.8 1.0 0.0 0.0 /\n"
SWOF = "SWOF\n 0.2 0.0 1.0 0.0\n 0.5 0.3 0.3 0.0\n 0.9 0.8 0.0 0.0 /\n"


def deck(body: str) -> DeckFile:
    return DeckFile("METRIC\n" + body + "\n")


def test_gas_oil_only_deck_is_rejected_by_default():
    with pytest.raises(ValidationError, match="allow_two_phase"):
        with pytest.warns(UserWarning):
            SatFunc.from_deck(deck(SGOF), dtype=np.float64)


def test_gas_oil_only_deck_builds_a_two_phase_table_when_allowed():
    satfunc = SatFunc.from_deck(deck(SGOF), allow_two_phase=True, dtype=np.float64)
    relperm = satfunc[1].relative_permeability
    assert isinstance(relperm, TwoPhaseRelPermTable)
    result = relperm.evaluate(0.2, 0.5, 0.3)
    assert float(result["water"]) == 0.0
    assert float(result["gas"]) > 0.0
    assert float(result["oil"]) > 0.0


def test_oil_water_only_deck_builds_a_two_phase_table_when_allowed():
    satfunc = SatFunc.from_deck(deck(SWOF), allow_two_phase=True, dtype=np.float64)
    relperm = satfunc[1].relative_permeability
    assert isinstance(relperm, TwoPhaseRelPermTable)
    result = relperm.evaluate(0.5, 0.5, 0.0)
    assert float(result["gas"]) == 0.0
    assert float(result["water"]) > 0.0


def test_complete_deck_is_unaffected_by_the_flag():
    satfunc = SatFunc.from_deck(deck(SWOF + SGOF), allow_two_phase=True, dtype=np.float64)
    assert isinstance(satfunc[1].relative_permeability, ThreePhaseRelPermTable)


def test_auto_family_prefers_the_family_that_is_complete():
    systems = detect_saturation_systems(deck(SGOF + SWFN_SOF3), 1)
    assert systems.family == "second"
    assert systems.oil_water and systems.gas_oil


SWFN_SOF3 = (
    "SWFN\n 0.2 0.0 0.0\n 0.9 0.8 0.0 /\n"
    "SGFN\n 0.0 0.0 0.0\n 0.8 1.0 0.0 /\n"
    "SOF3\n 0.0 0.0 0.0\n 0.5 0.3 0.3\n 0.8 1.0 1.0 /\n"
)
