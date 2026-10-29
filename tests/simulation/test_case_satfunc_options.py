"""Saturation-function options that `SimulationCase.from_deck` forwards to `SatFunc.from_deck`."""

import re
import warnings
from pathlib import Path

import pytest

import bores.simulation.case as case_module
from bores.blackoil.satfunc.relperm.tables import TwoPhaseRelPermTable
from bores.deck import DeckFile
from bores.errors import CaseLoadError, ValidationError
from bores.simulation.case import SimulationCase

SPE1_DECK = Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA"


@pytest.fixture(scope="module")
def gas_oil_only_deck(tmp_path_factory) -> DeckFile:
    """SPE1 without its oil-water table, so that it defines only the gas-oil system."""
    text = SPE1_DECK.read_text(encoding="utf-8")
    without_oil_water = re.sub(r"^SWOF\s*\n.*?/\s*\n", "", text, count=1, flags=re.M | re.S)
    assert without_oil_water != text
    path = tmp_path_factory.mktemp("deck") / "spe1_gas_oil_only.DATA"
    path.write_text(without_oil_water, encoding="utf-8")
    return DeckFile(path, encoding="utf-8")


@pytest.fixture
def recorded_satfunc_options(monkeypatch):
    """The keyword arguments `SatFunc.from_deck` receives; the real one still runs."""
    received = {}
    real = case_module.SatFunc.from_deck

    def recording(*args, **kwargs):
        received.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(case_module.SatFunc, "from_deck", recording)
    return received


def test_a_deck_with_one_system_is_rejected_by_default(gas_oil_only_deck):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(CaseLoadError) as raised:
            SimulationCase.from_deck(gas_oil_only_deck, temperature=200.0)
    assert isinstance(raised.value.__cause__, ValidationError)
    assert "allow_two_phase" in str(raised.value.__cause__)


def test_the_option_is_off_unless_asked_for(recorded_satfunc_options):
    deck_file = DeckFile(SPE1_DECK, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SimulationCase.from_deck(deck_file, temperature=200.0)
    assert recorded_satfunc_options["allow_two_phase"] is False


def test_the_option_reaches_the_saturation_functions(gas_oil_only_deck, recorded_satfunc_options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SimulationCase.from_deck(
            gas_oil_only_deck, temperature=200.0, allow_two_phase_satfunc=True
        )
    assert recorded_satfunc_options["allow_two_phase"] is True


def test_a_deck_with_one_system_loads_with_a_two_phase_table_when_allowed(
    gas_oil_only_deck, monkeypatch
):
    built = []
    real = case_module.SatFunc.from_deck

    def recording(*args, **kwargs):
        satfunc = real(*args, **kwargs)
        built.append(satfunc)
        return satfunc

    monkeypatch.setattr(case_module.SatFunc, "from_deck", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = SimulationCase.from_deck(
            gas_oil_only_deck, temperature=200.0, allow_two_phase_satfunc=True
        )
    assert case is not None
    (satfunc,) = built
    assert isinstance(satfunc[1].relative_permeability, TwoPhaseRelPermTable)
