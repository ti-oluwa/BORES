"""An explicit oil saturation (`SOIL`) with `SWAT` and `SGAS`: any two fix the third."""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import CaseLoadError
from bores.simulation.case import load_case

SPE1 = Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA"
CELLS = 300


def deck_with(solution: str) -> DeckFile:
    """SPE1 with its `EQUIL` replaced by `solution`, a block of explicit arrays."""
    text = SPE1.read_text()
    replaced, count = re.subn(r"(?ms)^EQUIL\b.*?(?=^SUMMARY)", solution + "\n", text, count=1)
    assert count == 1
    return DeckFile(replaced)


def explicit(water: float | None, gas: float | None, oil: float | None) -> str:
    arrays = {"PRESSURE": 4000.0, "RS": 1.0, "RV": 0.0, "SWAT": water, "SGAS": gas, "SOIL": oil}
    return "\n".join(
        f"{name}\n {CELLS}*{value} /\n" for name, value in arrays.items() if value is not None
    )


def load(solution: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_case(deck_with(solution), temperature=200.0).initial_state


def test_water_and_oil_give_the_gas_saturation():
    state = load(explicit(water=0.25, gas=None, oil=0.75))
    np.testing.assert_allclose(state.gas_saturation, 0.0, atol=1e-12)
    np.testing.assert_allclose(state.oil_saturation, 0.75)


def test_gas_and_oil_give_the_water_saturation():
    state = load(explicit(water=None, gas=0.05, oil=0.7))
    np.testing.assert_allclose(state.water_saturation, 0.25)
    np.testing.assert_allclose(state.oil_saturation, 0.7)


def test_all_three_that_agree_load():
    state = load(explicit(water=0.25, gas=0.05, oil=0.7))
    np.testing.assert_allclose(state.oil_saturation, 0.7)


def test_all_three_that_disagree_are_rejected_naming_the_discrepancy():
    with pytest.raises(CaseLoadError) as raised:
        load(explicit(water=0.25, gas=0.05, oil=0.5))
    assert "must add up to 1.0" in str(raised.value.__cause__)


def test_an_oil_saturation_that_leaves_a_negative_phase_is_rejected():
    with pytest.raises(CaseLoadError) as raised:
        load(explicit(water=0.6, gas=None, oil=0.6))
    assert "would be negative" in str(raised.value.__cause__)


def test_without_soil_oil_is_still_what_water_and_gas_leave():
    state = load(explicit(water=0.25, gas=0.05, oil=None))
    np.testing.assert_allclose(state.oil_saturation, 0.7)


def test_soil_alone_is_ignored_with_a_warning_when_equilibration_is_available():
    text = SPE1.read_text().replace("\nEQUIL", f"\nSOIL\n {CELLS}*0.5 /\n\nEQUIL", 1)
    with pytest.warns(UserWarning, match="oil saturation was given without"):
        state = load_case(DeckFile(text), temperature=200.0).initial_state
    assert not np.allclose(state.oil_saturation, 0.5)
