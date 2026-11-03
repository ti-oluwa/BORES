"""Output and file keywords, `DRSDT`/`DRVDT`, and when equilibration is needed."""

import warnings

import pytest

from bores.deck import DeckFile
from bores.initialization import initial_state_needs_equilibration
from bores.simulation.deck import load_run_spec
from bores.simulation.options import (
    MessageLimits,
    OutputOptions,
    PhysicsOptions,
    get_scheduled_at_start,
    load_dissolution_limits,
    load_output_options,
)

HEAD = "FIELD\nSPECGRID\n 2 2 1 1 F /\n"


def deck(body: str = "") -> DeckFile:
    return DeckFile(HEAD + body)


class TestOutputOptions:
    def test_a_deck_with_none_of_them_keeps_the_defaults(self):
        assert load_output_options(deck()) == OutputOptions()

    def test_the_file_and_echo_flags(self):
        options = load_output_options(
            deck("NOECHO\nFMTIN\nFMTOUT\nUNIFIN\nUNIFOUT\nINIT\nRUNSUM\n")
        )
        assert not options.echo_input
        assert options.formatted_input and options.formatted_output
        assert options.unified_input and options.unified_output
        assert options.write_initial_file and options.write_run_summary

    def test_gridfile_items_are_separate(self):
        options = load_output_options(deck("GRIDFILE\n 0 1 /\n"))
        assert not options.write_grid_file and options.write_extended_grid_file

    def test_messages_become_named_limits(self):
        limits = load_output_options(
            deck("MESSAGES\n 1 2 3 4 5 6 7 8 9 10 11 12 13 /\n")
        ).message_limits
        assert isinstance(limits, MessageLimits)
        assert (limits.message_print_limit, limits.warning_print_limit) == (1, 3)
        assert (limits.error_stop_limit, limits.group_print_limit) == (11, 13)
        assert load_output_options(deck()).message_limits is None

    def test_report_requests_keep_their_time_and_mnemonics(self):
        options = load_output_options(
            deck("RPTRST\n BASIC=2 /\nTSTEP\n 10 /\nRPTSCHED\n 'PRES' 'SGAS' /\n")
        )
        by_keyword = {request.keyword: request for request in options.report_requests}
        assert by_keyword["RPTRST"].time == 0.0 and by_keyword["RPTRST"].mnemonics == (
            ("BASIC", 2.0),
        )
        assert by_keyword["RPTSCHED"].time == 10.0
        assert [name for name, _ in by_keyword["RPTSCHED"].mnemonics] == ["PRES", "SGAS"]
        assert [request.time for request in options.report_requests] == [0.0, 10.0]

    def test_the_run_spec_carries_them(self):
        assert not load_run_spec(deck("NOECHO\n")).output_options.echo_input


class TestDissolutionLimits:
    def test_drsdt_and_drvdt_at_the_start(self):
        settings = load_dissolution_limits(deck("DRSDT\n 0.5 FREE /\nDRVDT\n 0.0001 /\n"))
        assert settings == {
            "maximum_solution_gor_increase_rate": 0.5,
            "solution_gor_limit_applies_to_all_cells": False,
            "maximum_vaporized_oil_gas_ratio_increase_rate": 0.0001,
        }

    def test_the_default_option_applies_the_limit_everywhere(self):
        assert load_dissolution_limits(deck("DRSDT\n 2 /\n"))[
            "solution_gor_limit_applies_to_all_cells"
        ]

    def test_a_defaulted_rate_means_no_limit(self):
        assert "maximum_solution_gor_increase_rate" not in load_dissolution_limits(
            deck("DRSDT\n /\n")
        )

    def test_a_deck_with_neither_sets_nothing(self):
        assert load_dissolution_limits(deck()) == {}
        # Only the regime constant, which every deck sets to Eclipse's 0
        assert load_run_spec(deck()).physics_options == PhysicsOptions(
            well_index_regime_constant=0.0
        )

    def test_later_records_are_warned_about_and_not_used(self):
        body = "DRSDT\n 0.5 /\nTSTEP\n 10 /\nDRSDT\n 9 /\nTSTEP\n 5 /\n"
        with pytest.warns(UserWarning, match="`DRSDT` appears again at 10"):
            settings = load_dissolution_limits(deck(body))
        assert settings["maximum_solution_gor_increase_rate"] == 0.5

    def test_the_rates_must_not_be_negative(self):
        with pytest.raises(ValueError):
            PhysicsOptions(maximum_solution_gor_increase_rate=-1.0)

    def test_the_loaded_values_reach_the_run_spec(self):
        spec = load_run_spec(deck("DRSDT\n 0.5 /\nTSTEP\n 5 /\n"))
        assert spec.physics_options.maximum_solution_gor_increase_rate == 0.5

    def test_a_scheduled_keyword_with_nothing_at_the_start_gives_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_scheduled_at_start(deck("TSTEP\n 5 /\nDRSDT\n 1 /\n"), "DRSDT") is None


class TestEquilibrationOnlyWhenNeeded:
    ARRAYS = "PRESSURE\n 4*3000 /\nRS\n 4*0.5 /\nRV\n 4*0 /\n"

    def test_a_deck_that_gives_everything_needs_none(self):
        assert not initial_state_needs_equilibration(
            deck(self.ARRAYS + "SWAT\n 4*0.2 /\nSGAS\n 4*0 /\n")
        )

    def test_oil_saturation_can_stand_in_for_one_of_the_two(self):
        assert not initial_state_needs_equilibration(
            deck(self.ARRAYS + "SWAT\n 4*0.2 /\nSOIL\n 4*0.8 /\n")
        )
        assert not initial_state_needs_equilibration(
            deck(self.ARRAYS + "SGAS\n 4*0 /\nSOIL\n 4*0.8 /\n")
        )

    def test_one_saturation_alone_is_not_enough(self):
        assert initial_state_needs_equilibration(deck(self.ARRAYS + "SWAT\n 4*0.2 /\n"))
        assert initial_state_needs_equilibration(deck(self.ARRAYS + "SOIL\n 4*0.8 /\n"))

    def test_a_missing_pressure_or_ratio_needs_it(self):
        saturations = "SWAT\n 4*0.2 /\nSGAS\n 4*0 /\n"
        assert initial_state_needs_equilibration(deck("RS\n 4*0.5 /\nRV\n 4*0 /\n" + saturations))
        assert initial_state_needs_equilibration(
            deck("PRESSURE\n 4*3000 /\nRV\n 4*0 /\n" + saturations)
        )
        assert initial_state_needs_equilibration(deck())
