"""Generic phases, run-spec options and dimensions, MINPV, TUNING names and later TUNING changes, and the presence constants."""

import warnings

import numpy as np
import pytest

from bores import correlations
from bores.constants import c
from bores.deck import DeckFile
from bores.deck.keywords.schedule import TUNING
from bores.deck.phases import DeckPhases, read_deck_phases
from bores.errors import ActionError, ValidationError
from bores.initialization import warn_on_inconsistent_phase_state
from bores.schedule.base import ScheduleContext
from bores.simulation.deck import load_run_spec
from bores.simulation.options import (
    load_equilibration_options,
    load_minimum_pore_volume,
    load_model_dimensions,
    load_saturation_options,
)
from bores.simulation.spec import RunSpec
from bores.timestep.control import RETUNABLE_SETTINGS, Timer
from bores.timestep.deck import (
    TUNING_DEFAULTS,
    get_tuning_at_start,
    load_timer,
    load_timer_schedule,
)
from bores.timestep.schedule import TIMER_CONTEXT_KEY, RetuneTimer
from bores.types import Phases, UnitSystem

HEAD = "METRIC\nSPECGRID\n 2 2 1 1 F /\n"
FIELD_HEAD = "FIELD\nSPECGRID\n 2 2 1 1 F /\n"


def deck(body: str = "", head: str = HEAD) -> DeckFile:
    return DeckFile(head + body)


class TestGenericPhases:
    def test_a_deck_phases_is_a_phases_and_can_drop_the_deck_labelling(self):
        deck_phases = read_deck_phases(deck("OIL\nWATER\nGAS\nDISGAS\n"))
        assert isinstance(deck_phases, DeckPhases) and isinstance(deck_phases, Phases)
        generic = deck_phases.to_phases()
        assert type(generic) is Phases and not hasattr(generic, "aqueous_slot")
        assert generic.is_three_phase and generic.dissolved_gas_in_oil

    def test_the_run_spec_holds_the_generic_type(self):
        spec = load_run_spec(deck("OIL\nWATER\nGAS\n"))
        assert type(spec.phases) is Phases

    def test_a_co2_storage_deck_keeps_its_aqueous_slot_only_on_the_deck_type(self):
        deck_phases = read_deck_phases(deck("OIL\nGAS\nCO2STOR\n"))
        assert deck_phases.aqueous_slot == "oil" and deck_phases.co2_storage
        assert deck_phases.to_phases().co2_storage


class TestRunSpecOptions:
    def test_saturation_options(self):
        options = load_saturation_options(deck("SATOPTS\n 'HYSTER' 'DIRECT' 'SURFTENS' 'FOO' /\n"))
        assert options.directional_relative_permeability and options.surface_tension_dependence
        assert not options.irreversible_directional_relative_permeability
        assert options.unrecognized == {"FOO"}
        assert load_saturation_options(deck()) == load_saturation_options(deck("SATOPTS\n /\n"))

    def test_equilibration_options(self):
        options = load_equilibration_options(deck("EQLOPTS\n THPRES MOBILE /\n"))
        assert options.threshold_pressures and options.mobile_fluid_endpoint_correction
        assert not options.quiescent_start and not options.irreversible_threshold_pressures

    def test_dimensions_are_none_without_any_dimension_keyword_and_per_item_otherwise(self):
        assert load_model_dimensions(deck()) is None
        dims = load_model_dimensions(deck("WELLDIMS\n 5 10 /\nREGDIMS\n 22 3 1* 20 /\n"))
        assert (dims.wells, dims.connections_per_well) == (5, 10)
        assert (
            dims.fluid_in_place_regions,
            dims.fluid_in_place_region_sets,
            dims.flux_regions,
        ) == (22, 3, 20)
        assert dims.equilibration_regions is None

    def test_the_run_spec_carries_them(self):
        spec = load_run_spec(
            deck("SATOPTS\n HYSTER SURFTENS /\nEQLOPTS\n QUIESC /\nWELLDIMS\n 3 /\n")
        )
        assert spec.hysteresis_enabled and spec.saturation_options.surface_tension_dependence
        assert spec.equilibration_options.quiescent_start and spec.dimensions.wells == 3


class TestMinimumPoreVolume:
    def test_it_is_none_without_the_keyword(self):
        assert load_minimum_pore_volume(deck()) is None
        assert load_run_spec(deck()).minimum_pore_volume is None

    def test_reservoir_barrels_become_cubic_feet_for_field_decks(self):
        assert load_minimum_pore_volume(deck("MINPV\n 1 /\n", FIELD_HEAD)) == pytest.approx(
            c.BARRELS_TO_CUBIC_FEET
        )

    def test_metric_decks_need_no_conversion(self):
        assert load_minimum_pore_volume(deck("MINPV\n 0.5 /\n")) == pytest.approx(0.5)

    def test_it_converts_with_the_run_spec(self):
        spec = load_run_spec(deck("MINPV\n 1 /\n", FIELD_HEAD))
        assert spec.convert(UnitSystem.METRIC).minimum_pore_volume == pytest.approx(0.158987294928)

    def test_a_negative_value_is_rejected(self):
        with pytest.raises(ValueError):
            RunSpec(unit_system=UnitSystem.METRIC, minimum_pore_volume=-1.0)

    def test_the_flow_mask_compares_reference_pore_volumes_and_keeps_all_cells_without_a_minimum(
        self,
    ):
        from bores.reservoir.model import Reservoir

        class Stub:
            pore_volumes = np.array([0.0, 1e-7, 1e-6, 5.0])
            grid = type("Grid", (), {"n_cells": 4})()

        mask = Reservoir.compute_flow_mask(Stub(), 1e-6)  # type: ignore[arg-type]
        assert mask.tolist() == [False, False, True, True]
        assert Reservoir.compute_flow_mask(Stub(), None).all()  # type: ignore[arg-type]


class TestTuningNames:
    def test_fields_have_descriptive_names(self):
        names = [field.name for _, fields in TUNING.records for field in fields]
        assert "initial_time_step" in names and "maximum_newton_iterations" in names
        assert not {"tsinit", "newtmx", "mxwsit"} & set(names)
        assert all(len(name) > 8 for name in names)

    def test_defaults_come_from_the_keyword_itself(self):
        for _, fields in TUNING.records:
            for field in fields:
                assert TUNING_DEFAULTS[field.name] == field.default

    def test_parse_uses_the_new_names(self):
        tuning = get_tuning_at_start(deck("TUNING\n 0.5 20 /\n /\n 15 /\n"))
        assert tuning["time_step"]["initial_time_step"] == 0.5
        assert tuning["iteration"]["maximum_newton_iterations"] == 15


def make_timer(**settings) -> Timer:
    return Timer(
        initial_step_size=1.0,
        maximum_step_size=30.0,
        minimum_step_size=0.1,
        simulation_time=100.0,
        unit_system=UnitSystem.METRIC,
        **settings,
    )


class TestRetuneTimer:
    def test_settings_change_and_the_planned_step_stays_inside_the_new_limits(self):
        timer = make_timer()
        timer.next_step_size = 25.0
        timer.retune({"maximum_step_size": 10.0, "maximum_growth_per_step": 2.0})
        assert timer.maximum_step_size == 10.0 and timer.maximum_growth_per_step == 2.0
        assert timer.next_step_size == 10.0

    def test_an_initial_step_caps_the_next_step(self):
        timer = make_timer()
        timer.next_step_size = 20.0
        timer.retune({"initial_step_size": 4.0})
        assert timer.next_step_size == 4.0

    def test_a_failure_ceiling_the_new_maximum_has_passed_is_dropped(self):
        timer = make_timer()
        timer.failure_ceiling = 12.0
        timer.retune({"maximum_step_size": 10.0})
        assert timer.failure_ceiling is None

    def test_settings_that_do_not_hold_together_change_nothing(self):
        timer = make_timer()
        with pytest.raises(ValidationError):
            timer.retune({"backoff_factor": 2.0})
        assert timer.backoff_factor == 0.5
        with pytest.raises(ValidationError):
            timer.retune({"minimum_step_size": 50.0})
        assert timer.minimum_step_size == 0.1 and timer.initial_step_size == 1.0

    def test_a_setting_that_cannot_change_during_a_run_is_refused(self):
        with pytest.raises(ValidationError, match="simulation_time"):
            make_timer().retune({"simulation_time": 5.0})

    def test_the_action_changes_the_timer_in_the_context_and_returns_the_model(self):
        timer = make_timer()
        model = object()
        context = ScheduleContext(time=10.0, extra={TIMER_CONTEXT_KEY: timer})
        assert RetuneTimer(settings={"maximum_step_size": 5.0})(model, context) is model
        assert timer.maximum_step_size == 5.0

    def test_the_action_needs_a_timer_in_the_context(self):
        with pytest.raises(ActionError, match="needs the run's `Timer`"):
            RetuneTimer(settings={"maximum_step_size": 5.0})(object(), ScheduleContext(time=1.0))

    def test_every_retunable_setting_is_a_timer_attribute(self):
        timer = make_timer()
        assert all(hasattr(timer, name) for name in RETUNABLE_SETTINGS)


class TestTimerSchedule:
    BODY = (
        "TUNING\n 1 /\n /\n /\nTSTEP\n 10 /\n"
        "TUNING\n 2 7 /\n /\n /\nTSTEP\n 5 /\n"
        "TUNING\n /\n /\n /\nTSTEP\n 5 /\n"
    )

    def test_each_later_tuning_that_sets_something_is_one_item_at_its_time(self):
        items = load_timer_schedule(deck(self.BODY))
        assert [item.name for item in items] == ["tuning at 10"]
        assert items[0].event.at == 10.0
        assert dict(items[0].action.settings) == {
            "initial_step_size": 2.0,
            "maximum_step_size": 7.0,
        }

    def test_a_deck_with_one_tuning_or_none_has_nothing_scheduled(self):
        assert load_timer_schedule(deck("TUNING\n 1 /\n /\n /\nTSTEP\n 10 /\n")) == ()
        assert load_timer_schedule(deck("TSTEP\n 10 /\n")) == ()

    def test_the_start_timer_warns_once_that_later_changes_exist(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_timer(deck(self.BODY))
        assert len(caught) == 1 and "load_timer_schedule" in str(caught[0].message)

    def test_a_scheduled_change_reaches_a_real_timer(self):
        items = load_timer_schedule(deck(self.BODY))
        timer = load_timer(deck("TUNING\n 1 /\n /\n /\nTSTEP\n 10 /\n"))
        context = ScheduleContext(time=10.0, previous_time=0.0, extra={TIMER_CONTEXT_KEY: timer})
        assert items[0].event(object(), context)
        items[0].action(object(), context)
        assert timer.maximum_step_size == 7.0


class TestPresenceConstants:
    def state(self, **options):
        pressure = np.array([100.0, 100.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_on_inconsistent_phase_state(
                pressure=pressure,
                oil_bubble_point_pressure=np.array([200.0, 50.0]),
                gas_dew_point_pressure=np.array([0.0, 0.0]),
                water_saturation=np.array([0.2, 0.2]),
                gas_saturation=np.array([0.0, 0.0]),
                oil_saturation=np.array([0.8, 0.8]),
                active=None,
                explicit_fields=[],
                **options,
            )
        return caught

    def test_the_constants_are_registered_with_the_values_that_were_hardcoded(self):
        assert c.OIL_PRESENCE_SATURATION == 1e-3 and c.GAS_PRESENCE_SATURATION == 1e-3
        assert c.FREE_PHASE_SATURATION == 1e-6 and c.PRESSURE_CONSISTENCY_TOLERANCE == 1e-4

    def test_the_defaults_come_from_the_constants(self):
        assert len(self.state()) == 1

    def test_arguments_override_the_constants(self):
        assert len(self.state(oil_presence_saturation=0.9)) == 0  # 0.8 oil no longer counts as oil
        assert len(self.state(pressure_consistency_tolerance=2.0)) == 0  # pressure floor 1 - 2 < 0

    def test_the_run_spec_fields_default_to_none_so_the_constants_apply(self):
        spec = RunSpec(unit_system=UnitSystem.METRIC)
        assert spec.oil_presence_saturation is None and spec.gas_presence_saturation is None
        assert spec.pressure_consistency_tolerance is None
        with pytest.raises(ValueError):
            RunSpec(unit_system=UnitSystem.METRIC, oil_presence_saturation=-0.1)


def test_api_gravity_converts_to_specific_gravity_and_back():
    specific_gravity = correlations.compute_oil_specific_gravity_from_api_gravity(
        oil_api_gravity=35.0
    )
    assert correlations.compute_oil_api_gravity(specific_gravity) == pytest.approx(35.0)
