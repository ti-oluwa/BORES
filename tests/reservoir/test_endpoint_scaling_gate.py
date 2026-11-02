"""End point arrays are used only when end point scaling is on, and the five extra end points."""

import warnings

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.reservoir.rock.model import ENDPOINT_SCALING_KEYWORDS, Rock
from bores.simulation.options import load_saturation_options

BASE = "METRIC\nSPECGRID\n 2 1 1 1 F /\nPORO\n 0.2 0.2 /\nPERMX\n 100 100 /\n"
ARRAYS = "SWL\n 0.04 0.16 /\nSWCR\n 0.12 0.24 /\nSGCR\n 0.03 0.05 /\n"


class Grid:
    n_cells = 2


def rock(body: str, **options) -> Rock:
    return Rock.from_deck(DeckFile(BASE + body), grid=Grid(), **options)  # type: ignore[arg-type]


class TestGate:
    def test_arrays_without_endscale_are_ignored_with_a_warning(self):
        with pytest.warns(
            UserWarning, match="`SWL`, `SWCR`, `SGCR` are in the deck but end point"
        ):
            result = rock(ARRAYS)
        np.testing.assert_allclose(result.connate_water_saturation, 0.0)
        np.testing.assert_allclose(result.residual_gas_saturation, 0.0)

    def test_the_warning_names_only_what_is_present(self):
        with pytest.warns(UserWarning, match="`SWL` is in the deck"):
            rock("SWL\n 0.04 0.16 /\n")

    def test_arrays_with_endscale_are_used_and_nothing_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rock("ENDSCALE\n /\n" + ARRAYS)
        np.testing.assert_allclose(result.connate_water_saturation, [0.04, 0.16])
        np.testing.assert_allclose(result.residual_gas_saturation, [0.03, 0.05])

    def test_a_deck_with_no_arrays_and_no_endscale_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rock("")

    def test_the_argument_overrides_the_deck_both_ways(self):
        with pytest.warns(UserWarning):
            off = rock("ENDSCALE\n /\n" + ARRAYS, endpoint_scaling=False)
        np.testing.assert_allclose(off.connate_water_saturation, 0.0)
        on = rock(ARRAYS, endpoint_scaling=True)
        np.testing.assert_allclose(on.connate_water_saturation, [0.04, 0.16])

    def test_every_gated_keyword_is_one_the_rock_reads(self):
        assert {
            "SWL",
            "SWCR",
            "SGCR",
            "SOWCR",
            "SOGCR",
            "SGL",
            "SGU",
            "SWU",
            "SWLPC",
            "SGLPC",
        } <= set(ENDPOINT_SCALING_KEYWORDS)

    def test_the_run_spec_flag_follows_the_deck_unless_set(self):
        assert (
            load_saturation_options(DeckFile(BASE + "ENDSCALE\n /\n")).endpoint_scaling_enabled
            is True
        )
        assert load_saturation_options(DeckFile(BASE)).endpoint_scaling_enabled is False
        from bores.simulation.options import SaturationOptions

        assert SaturationOptions().endpoint_scaling_enabled is None


class TestNewEndpoints:
    def test_they_are_none_when_the_deck_gives_none(self):
        result = rock("ENDSCALE\n /\n" + ARRAYS)
        assert result.connate_gas_saturation is None and result.maximum_gas_saturation is None
        assert result.maximum_water_saturation is None
        assert result.capillary_connate_water_saturation is None
        assert result.capillary_connate_gas_saturation is None

    def test_they_are_read(self):
        result = rock(
            "ENDSCALE\n /\nSWL\n 0.1 0.2 /\nSGL\n 0.01 0.02 /\nSGU\n 0.8 0.7 /\nSWU\n 0.95 0.9 /\n"
            "SWLPC\n 0.11 0.21 /\nSGLPC\n 0.012 0.022 /\n"
        )
        np.testing.assert_allclose(result.connate_gas_saturation, [0.01, 0.02])
        np.testing.assert_allclose(result.maximum_gas_saturation, [0.8, 0.7])
        np.testing.assert_allclose(result.maximum_water_saturation, [0.95, 0.9])
        np.testing.assert_allclose(result.capillary_connate_water_saturation, [0.11, 0.21])
        np.testing.assert_allclose(result.capillary_connate_gas_saturation, [0.012, 0.022])

    def test_a_cell_the_deck_does_not_set_takes_the_value_the_functions_would_give(self):
        body = (
            "ENDSCALE\n /\nSWL\n 0.1 0.2 /\n"
            "EQUALS\n SGL 0.01 1 1 1 1 1 1 /\n SGU 0.8 1 1 1 1 1 1 /\n SWU 0.9 1 1 1 1 1 1 /\n"
            " SWLPC 0.15 1 1 1 1 1 1 /\n SGLPC 0.02 1 1 1 1 1 1 /\n/\n"
        )
        result = rock(body)
        np.testing.assert_allclose(result.connate_gas_saturation, [0.01, 0.0])
        np.testing.assert_allclose(result.maximum_gas_saturation, [0.8, 1.0 - 0.2])
        np.testing.assert_allclose(result.maximum_water_saturation, [0.9, 1.0])
        np.testing.assert_allclose(result.capillary_connate_water_saturation, [0.15, 0.2])
        np.testing.assert_allclose(result.capillary_connate_gas_saturation, [0.02, 0.0])

    def test_a_saturation_outside_zero_to_one_is_rejected(self):
        with pytest.raises(ValidationError, match="maximum_water_saturation"):
            rock("ENDSCALE\n /\nSWU\n 1.5 1 /\n")

    def test_they_survive_unit_conversion_unchanged(self):
        from bores.types import UnitSystem

        result = rock("ENDSCALE\n /\nSGL\n 0.01 0.02 /\n")
        converted = result.convert(UnitSystem.FIELD)
        np.testing.assert_allclose(converted.connate_gas_saturation, [0.01, 0.02])
