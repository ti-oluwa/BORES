"""Tests for translating `SATOPTS` and `EHYSTR` into `HysteresisOptions`."""

import pytest

from bores.blackoil.satfunc.hysteresis import (
    LIMITING_CURVES_MAP,
    RELPERM_HYSTERESIS_MODEL_MAP,
    HysteresisOptions,
)
from bores.constants import c
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.types import FluidPhase, RelPermHysteresisModel


def _deck(*, satopts: str = " HYSTER /", ehystr: str | None = None) -> DeckFile:
    text = f"FIELD\nRUNSPEC\nDIMENS\n 1 1 1 /\nSATOPTS\n{satopts}\nPROPS\n"
    if ehystr is not None:
        text += f"EHYSTR\n {ehystr} /\n"
    return DeckFile(text)


def test_a_deck_without_the_hyster_option_has_no_hysteresis():
    assert HysteresisOptions.from_deck(_deck(satopts=" /")) is None
    assert (
        HysteresisOptions.from_deck(DeckFile("FIELD\nRUNSPEC\nDIMENS\n 1 1 1 /\nPROPS\n")) is None
    )


def test_hyster_without_ehystr_gives_the_eclipse_defaults():
    options = HysteresisOptions.from_deck(_deck())
    assert options == HysteresisOptions()
    assert options.relperm_model == RelPermHysteresisModel.CARLSON
    assert not options.wetting_phase_uses_imbibition
    assert options.capillary_pressure_curvature == pytest.approx(
        c.HYSTERESIS_CAPILLARY_PRESSURE_CURVATURE
    )
    assert options.killough_wetting_curvature == pytest.approx(
        c.HYSTERESIS_KILLOUGH_WETTING_CURVATURE
    )
    assert options.killough_trapped_modifier == pytest.approx(
        c.HYSTERESIS_KILLOUGH_TRAPPED_MODIFIER
    )
    assert options.apply_to_relperm and options.apply_to_capillary_pressure
    assert options.wetting_phase == FluidPhase.OIL


@pytest.mark.parametrize(
    ("code", "model", "wetting_phase_uses_imbibition"),
    [
        (-1, RelPermHysteresisModel.NONE, False),
        (0, RelPermHysteresisModel.CARLSON, False),
        (1, RelPermHysteresisModel.CARLSON, True),
        (2, RelPermHysteresisModel.KILLOUGH, False),
        (3, RelPermHysteresisModel.KILLOUGH, True),
    ],
)
def test_ehystr_item_two_selects_the_model_and_the_wetting_phase_curve(
    code, model, wetting_phase_uses_imbibition
):
    options = HysteresisOptions.from_deck(_deck(ehystr=f"0.1 {code}"))
    assert options.relperm_model == model
    assert options.wetting_phase_uses_imbibition is wetting_phase_uses_imbibition


def test_an_unknown_model_code_is_rejected():
    assert set(RELPERM_HYSTERESIS_MODEL_MAP) == {-1, 0, 1, 2, 3}
    with pytest.raises(ValidationError):
        HysteresisOptions.from_deck(_deck(ehystr="0.1 7"))


@pytest.mark.parametrize(
    ("limiting_curves", "relperm", "capillary_pressure"),
    [("BOTH", True, True), ("KR", True, False), ("PC", False, True)],
)
def test_limiting_curves_switch_relperm_and_capillary_pressure_hysteresis(
    limiting_curves, relperm, capillary_pressure
):
    options = HysteresisOptions.from_deck(_deck(ehystr=f"0.1 0 1.0 0.1 {limiting_curves}"))
    assert (options.apply_to_relperm, options.apply_to_capillary_pressure) == (
        relperm,
        capillary_pressure,
    )
    assert set(LIMITING_CURVES_MAP) == {"BOTH", "KR", "PC"}


def test_every_item_reaches_its_own_field():
    options = HysteresisOptions.from_deck(
        _deck(ehystr="0.25 3 1.5 0.2 KR NEW IMBIB WATER YES NO YES 0.05 1 1")
    )
    assert options.capillary_pressure_curvature == pytest.approx(0.25)
    assert options.relperm_model == RelPermHysteresisModel.KILLOUGH
    assert options.wetting_phase_uses_imbibition
    assert options.killough_wetting_curvature == pytest.approx(1.5)
    assert options.killough_trapped_modifier == pytest.approx(0.2)
    assert (options.apply_to_relperm, options.apply_to_capillary_pressure) == (True, False)
    assert not options.retrace_capillary_pressure
    assert options.starts_on_imbibition_curve
    assert options.wetting_phase == FluidPhase.WATER
    assert options.baker_interpolation_oil
    assert not options.baker_interpolation_gas
    assert options.baker_interpolation_water
    assert options.threshold_saturation == pytest.approx(0.05)
    assert options.fix_killough_wetting_phase
    assert options.scale_capillary_pressure


def test_the_blended_model_is_chosen_explicitly_and_the_deck_still_sets_the_rest():
    options = HysteresisOptions.from_deck(
        _deck(ehystr="0.3 3"), relperm_model=RelPermHysteresisModel.BLENDED_SCANNING
    )
    assert options.relperm_model == RelPermHysteresisModel.BLENDED_SCANNING
    assert options.wetting_phase_uses_imbibition
    assert options.capillary_pressure_curvature == pytest.approx(0.3)
    without_ehystr = HysteresisOptions.from_deck(
        _deck(), relperm_model=RelPermHysteresisModel.BLENDED_SCANNING
    )
    assert without_ehystr.relperm_model == RelPermHysteresisModel.BLENDED_SCANNING


def test_relperm_hysteresis_is_inactive_when_the_model_is_none_or_relperm_is_not_limited():
    assert HysteresisOptions().relperm_hysteresis_active
    assert not HysteresisOptions(
        relperm_model=RelPermHysteresisModel.NONE
    ).relperm_hysteresis_active
    assert not HysteresisOptions(apply_to_relperm=False).relperm_hysteresis_active


@pytest.mark.parametrize(
    ("wetting_phase", "water_increasing", "gas_increasing"),
    [
        (FluidPhase.OIL, False, False),
        (FluidPhase.WATER, True, False),
        (FluidPhase.GAS, False, True),
    ],
)
def test_the_wetting_phase_sets_which_way_each_phase_imbibes(
    wetting_phase, water_increasing, gas_increasing
):
    options = HysteresisOptions(wetting_phase=wetting_phase)
    assert options.water_imbibition_when_increasing is water_increasing
    assert options.gas_imbibition_when_increasing is gas_increasing


@pytest.mark.parametrize(
    "overrides",
    [
        {"capillary_pressure_curvature": -0.1},
        {"killough_wetting_curvature": 0.0},
        {"killough_trapped_modifier": -1.0},
        {"threshold_saturation": 1.0},
        {"wetting_phase": FluidPhase.LIQUID_OIL},
    ],
)
def test_out_of_range_options_are_rejected(overrides):
    with pytest.raises(ValidationError):
        HysteresisOptions(**overrides)
