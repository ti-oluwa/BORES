"""The grouped run options on `RunSpec`, their validation and unit conversion, and the extra run keywords."""

import attrs
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.simulation.deck import load_run_spec
from bores.simulation.options import (
    NewtonOptions,
    OutputOptions,
    PhaseOptions,
    PhysicsOptions,
    SolverOptions,
    StepLimitOptions,
    load_model_dimensions,
    load_saturation_options,
)
from bores.simulation.spec import RunSpec
from bores.types import UnitSystem
from bores.units import get_conversion_factors

HEAD = "METRIC\nSPECGRID\n 2 2 1 1 F /\n"
FIELD_HEAD = "FIELD\nSPECGRID\n 2 2 1 1 F /\n"

GROUPS = {
    "solver_options": SolverOptions,
    "newton_options": NewtonOptions,
    "step_limit_options": StepLimitOptions,
    "physics_options": PhysicsOptions,
    "phase_options": PhaseOptions,
    "output_options": OutputOptions,
}


class TestStructure:
    def test_every_group_is_a_run_spec_attribute_with_its_own_defaults(self):
        spec = RunSpec()
        for name, group in GROUPS.items():
            assert getattr(spec, name) == group()

    def test_the_flat_level_is_small(self):
        assert len(attrs.fields(RunSpec)) <= 20

    def test_no_setting_is_in_two_groups(self):
        names = [field.name for group in GROUPS.values() for field in attrs.fields(group)]
        assert len(names) == len(set(names))

    def test_the_defaults_are_the_ones_the_flat_fields_had(self):
        spec = RunSpec()
        assert spec.newton_options.maximum_newton_iterations == 10
        assert spec.phase_options.phase_appearance_tolerance == 1e-6
        assert spec.output_options.output_frequency == 1
        assert spec.solver_options.pressure_solver == "bicgstab"
        assert spec.physics_options.minimum_pore_volume is None

    def test_changing_one_group_leaves_the_others(self):
        spec = attrs.evolve(RunSpec(), newton_options=NewtonOptions(maximum_newton_iterations=20))
        assert spec.newton_options.maximum_newton_iterations == 20
        assert spec.solver_options == SolverOptions()


class TestValidationMovedWithTheFields:
    def test_a_newton_iteration_limit_outside_the_range_is_rejected(self):
        with pytest.raises(ValueError):
            NewtonOptions(maximum_newton_iterations=51)
        with pytest.raises(ValueError):
            NewtonOptions(maximum_newton_iterations=0)

    def test_an_output_frequency_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            OutputOptions(output_frequency=0)

    def test_injector_saturation_must_exceed_the_phase_appearance_tolerance(self):
        with pytest.raises(
            ValidationError, match="physics_options.minimum_injector_gas_saturation"
        ):
            RunSpec(
                phase_options=PhaseOptions(phase_appearance_tolerance=1e-3),
                physics_options=PhysicsOptions(minimum_injector_gas_saturation=1e-3),
            )
        RunSpec(
            phase_options=PhaseOptions(phase_appearance_tolerance=1e-3),
            physics_options=PhysicsOptions(minimum_injector_water_saturation=2e-3),
        )


class TestConversion:
    def test_the_pressure_change_limit_and_minimum_pore_volume_follow_the_unit_system(self):
        spec = RunSpec(
            unit_system=UnitSystem.FIELD,
            step_limit_options=StepLimitOptions(maximum_pressure_change=100.0),
            physics_options=PhysicsOptions(minimum_pore_volume=5.614583333333333),
        )
        metric = spec.convert(UnitSystem.METRIC)
        pressure_factor = get_conversion_factors(UnitSystem.FIELD, UnitSystem.METRIC)["pressure"]
        assert metric.step_limit_options.maximum_pressure_change == pytest.approx(
            100.0 * pressure_factor
        )
        assert metric.physics_options.minimum_pore_volume == pytest.approx(0.158987294928)
        assert metric.newton_options == spec.newton_options

    def test_a_group_with_nothing_to_convert_is_returned_as_it_is(self):
        physics = PhysicsOptions()
        assert physics.convert(volume_factor=2.0) is physics


class TestExtraRunKeywords:
    def test_endscale_and_scalecrs(self):
        options = load_saturation_options(
            DeckFile(HEAD + "ENDSCALE\n DIRECT IRREVERS 2 7 /\nSCALECRS\n YES /\n")
        )
        assert options.endpoint_scaling_enabled and options.directional_endpoint_scaling
        assert options.irreversible_endpoint_scaling and options.scale_critical_saturations

    def test_a_plain_endscale_turns_scaling_on_without_direction_or_irreversibility(self):
        options = load_saturation_options(DeckFile(HEAD + "ENDSCALE\n /\n"))
        assert options.endpoint_scaling_enabled
        assert (
            not options.directional_endpoint_scaling and not options.irreversible_endpoint_scaling
        )

    def test_a_deck_without_them_has_none_on(self):
        options = load_saturation_options(DeckFile(HEAD))
        assert not options.endpoint_scaling_enabled and not options.scale_critical_saturations
        assert not load_saturation_options(
            DeckFile(HEAD + "SCALECRS\n NO /\n")
        ).scale_critical_saturations

    def test_hysteresis_comes_from_satopts_and_lives_with_the_saturation_options(self):
        assert load_saturation_options(DeckFile(HEAD + "SATOPTS\n HYSTER /\n")).hysteresis_enabled
        assert not load_saturation_options(
            DeckFile(HEAD + "SATOPTS\n DIRECT /\n")
        ).hysteresis_enabled

    def test_nosim_means_data_check_only(self):
        assert load_run_spec(DeckFile(HEAD + "NOSIM\n")).data_check_only is True
        assert load_run_spec(DeckFile(HEAD)).data_check_only is False

    def test_the_extra_capacities_are_read(self):
        dims = load_model_dimensions(
            DeckFile(
                HEAD + "ENDSCALE\n DIRECT REVERS 2 7 /\nFAULTDIM\n 100 /\nPIMTDIMS\n 3 20 /\n"
            )
        )
        assert (dims.endpoint_depth_tables, dims.endpoint_depth_nodes) == (2, 7)
        assert dims.fault_segments == 100
        assert (dims.multiplier_tables, dims.multiplier_table_rows) == (3, 20)

    def test_a_deck_loads_a_run_spec_with_each_setting_in_its_group(self):
        spec = load_run_spec(
            DeckFile(
                FIELD_HEAD
                + "SATOPTS\n HYSTER /\nMINPV\n 1 /\nTUNING\n /\n /\n 20 /\nTSTEP\n 5 /\n"
            )
        )
        assert spec.saturation_options.hysteresis_enabled
        assert spec.physics_options.minimum_pore_volume == pytest.approx(5.614583333333333)
        assert spec.newton_options.maximum_newton_iterations == 20
