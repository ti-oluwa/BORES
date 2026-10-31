"""Loading the run configuration, timer, well control settings, phases-driven PVT, `GRAVITY`, `PORV` and `*DIMS` checks."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from bores.blackoil.pvt.regions import PVT, pvt_requirements
from bores.blackoil.pvt.static import load_density_records
from bores.deck import DeckFile
from bores.deck.dimensions import DeckDimensionWarning, check_deck_dimensions, warn_deck_dimensions
from bores.deck.phases import read_deck_phases
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.reservoir.rock.model import Rock
from bores.simulation.deck import load_run_spec
from bores.timestep.deck import load_timer
from bores.types import UnitSystem
from bores.wells.deck import load_well_control_spec
from bores.wells.resolution.spec import ConnectionPressureMode

SPE1 = Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA"
HEAD = "METRIC\nSPECGRID\n 2 2 1 1 F /\n"
WELLS = "WELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\nCOMPDAT\n 'P' 1 1 1 1 OPEN /\n/\n"


def deck(body: str = "") -> DeckFile:
    return DeckFile(HEAD + body)


class TestTimer:
    def test_schedule_length_and_default_step_sizes(self):
        timer = load_timer(deck("SCHEDULE\nTSTEP\n 10 20 /\n"))
        assert timer.simulation_time == 30.0
        assert (timer.initial_step_size, timer.maximum_step_size, timer.minimum_step_size) == (
            1.0,
            365.0,
            0.1,
        )
        assert timer.unit_system == UnitSystem.METRIC

    def test_tuning_step_sizes_are_read_and_initial_is_kept_inside_the_limits(self):
        timer = load_timer(deck("TUNING\n 50 20 2 /\n /\n /\nTSTEP\n 100 /\n"))
        assert (timer.initial_step_size, timer.maximum_step_size, timer.minimum_step_size) == (
            20.0,
            20.0,
            2.0,
        )

    def test_optional_settings_are_taken_only_when_the_deck_departs_from_eclipses_default(self):
        plain = load_timer(deck("TUNING\n /\n /\n /\nTSTEP\n 10 /\n"))
        tuned = load_timer(
            deck(
                "TUNING\n 1* 1* 1* 1* 2.0 1* 0.2 /\n 1* 1* 1* 1* 1* 1* 5e-6 /\n /\nTSTEP\n 10 /\n"
            )
        )
        assert plain.maximum_growth_per_step == 1.3 and plain.backoff_factor == 0.5
        assert plain.maximum_total_relative_mbe is None
        assert tuned.maximum_growth_per_step == 2.0
        assert (tuned.backoff_factor, tuned.aggressive_backoff_factor) == (0.2, 0.2)
        assert tuned.maximum_total_relative_mbe == pytest.approx(5e-6)

    def test_a_minimum_above_the_maximum_is_rejected(self):
        with pytest.raises(ValidationError, match="inconsistent"):
            load_timer(deck("TUNING\n 1 5 9 /\n /\n /\nTSTEP\n 10 /\n"))

    def test_no_schedule_length_gives_no_timer(self):
        assert load_timer(deck()) is None

    def test_a_later_tuning_is_reported_not_silently_used(self):
        body = "TUNING\n 1 /\n /\n /\nTSTEP\n 10 /\nTUNING\n 2 /\n /\n /\nTSTEP\n 5 /\n"
        with pytest.warns(UserWarning, match="appears again at 10"):
            timer = load_timer(deck(body))
        assert timer.initial_step_size == 1.0


class TestWellControlSpec:
    def test_deck_wells_use_hydrostatic_connection_pressures(self):
        assert (
            load_well_control_spec(deck(WELLS)).connection_pressure_mode
            == ConnectionPressureMode.HYDROSTATIC
        )
        assert (
            load_well_control_spec(deck()).connection_pressure_mode
            == ConnectionPressureMode.HYDRAULIC
        )

    def test_tuning_can_raise_the_iteration_limits_but_never_lower_them(self):
        higher = load_well_control_spec(deck("TUNING\n /\n /\n 1* 1* 1* 1* 30 50 /\n"))
        lower = load_well_control_spec(deck("TUNING\n /\n /\n 1* 1* 1* 1* 4 4 /\n"))
        assert (higher.max_fixed_point_iterations, higher.max_bisection_iterations) == (30, 50)
        assert (lower.max_fixed_point_iterations, lower.max_bisection_iterations) == (20, 40)


class TestRunSpec:
    def test_spe1(self):
        spec = load_run_spec(DeckFile(SPE1, encoding="utf-8"))
        assert spec.unit_system == UnitSystem.FIELD
        assert spec.start_date is not None and spec.start_date.year == 2015
        assert (
            spec.phases is not None
            and spec.phases.is_three_phase
            and spec.phases.dissolved_gas_in_oil
        )
        assert spec.timer is not None and spec.timer.unit_system == UnitSystem.FIELD
        assert spec.hysteresis_enabled is False
        assert spec.well_control.connection_pressure_mode == ConnectionPressureMode.HYDROSTATIC

    def test_hysteresis_comes_from_satopts(self):
        assert load_run_spec(deck("OIL\nGAS\nSATOPTS\n 'HYSTER' /\n")).hysteresis_enabled is True
        assert load_run_spec(deck("OIL\nGAS\nSATOPTS\n 'DIRECT' /\n")).hysteresis_enabled is False

    def test_newton_iterations_follow_a_non_default_newtmx_only(self):
        assert load_run_spec(deck("TUNING\n /\n /\n 20 /\n")).maximum_newton_iterations == 20
        assert load_run_spec(deck("TUNING\n /\n /\n /\n")).maximum_newton_iterations == 10
        assert load_run_spec(deck("TUNING\n /\n /\n 99 /\n")).maximum_newton_iterations == 10

    def test_a_deck_with_no_phases_gives_none(self):
        assert load_run_spec(deck()).phases is None


GAS_WATER_DECK = (
    "GAS\nWATER\nPVDG\n 1000 1.0 0.02\n 5000 0.5 0.03 /\nPVTW\n 3000 1.02 3e-6 0.5 0 /\n"
)


class TestPhasesDrivePvt:
    def test_requirements_follow_the_phases(self):
        three = read_deck_phases(deck("OIL\nWATER\nGAS\nDISGAS\n"))
        assert pvt_requirements(three) == {
            "all_of": ("PVTW",),
            "any_of": (("PVTO", "PVCO"), ("PVTG", "PVDG")),
        }
        gas_water = read_deck_phases(deck("GAS\nWATER\n"))
        assert pvt_requirements(gas_water) == {"all_of": ("PVTW",), "any_of": (("PVTG", "PVDG"),)}
        co2 = read_deck_phases(deck("GAS\nWATER\nCO2STORE\n"))
        assert pvt_requirements(co2) == {"all_of": (), "any_of": ()}

    def test_a_gas_water_deck_loads_with_no_oil_table(self):
        pvt = PVT.from_deck(deck(GAS_WATER_DECK), temperature=150.0)
        region = pvt.regions[1]
        assert region.tables.oil is None
        assert region.tables.gas is not None and region.tables.water is not None

    def test_a_table_the_phases_need_is_reported_before_parsing(self):
        with pytest.raises(ValidationError, match="`PVTW`"):
            PVT.from_deck(
                deck("GAS\nWATER\nPVDG\n 1000 1.0 0.02\n 5000 0.5 0.03 /\n"), temperature=150.0
            )

    def test_tables_of_an_undeclared_phase_are_left_unread(self):
        body = "OIL\nWATER\nPVDO\n 1000 1.1 1\n 5000 1.05 1.2 /\nPVTW\n 3000 1.02 3e-6 0.5 0 /\nPVDG\n 1 1 1 /\n"
        region = PVT.from_deck(deck(body), temperature=150.0).regions[1]
        assert region.tables.gas is None and region.tables.oil is not None

    def test_a_co2_storage_deck_is_refused_clearly(self):
        with pytest.raises(ValidationError, match="CO2 storage"):
            PVT.from_deck(deck("GAS\nWATER\nCO2STORE\n"), temperature=150.0)


class TestGravity:
    def test_gravity_becomes_densities_against_1000_water_and_1_22_air(self):
        record = load_density_records(deck("GRAVITY\n 45 1.0 0.8 /\n"))[0][0]
        assert record["oil"] == pytest.approx(141.5 / 176.5 * 1000.0)
        assert record["water"] == pytest.approx(1000.0)
        assert record["gas"] == pytest.approx(0.8 * 1.22)

    def test_density_wins_over_gravity_and_neither_gives_none(self):
        both = load_density_records(deck("GRAVITY\n 45 1.0 0.8 /\nDENSITY\n 800 1000 1 /\n"))
        assert both[0][0]["oil"] == 800.0
        assert load_density_records(deck()) is None

    def test_field_units_are_converted(self):
        record = load_density_records(
            DeckFile("FIELD\nSPECGRID\n 2 2 1 1 F /\nGRAVITY\n 45 1.0 0.8 /\n")
        )[0][0]
        assert record["water"] == pytest.approx(62.428, rel=1e-4)


def small_grid(body: str) -> tuple[DeckFile, Grid]:
    text = HEAD + "DX\n 4*10 /\nDY\n 4*10 /\nDZ\n 4*5 /\nTOPS\n 4*1000 /\nPERMX\n 4*100 /\n" + body
    deck_file = DeckFile(text)
    return deck_file, Grid.from_deck(deck_file)


class TestPorv:
    def test_porv_overrides_poro_through_the_porosity_that_reproduces_it(self):
        deck_file, grid = small_grid("PORO\n 4*0.2 /\nNTG\n 4*0.5 /\nPORV\n 100 200 250 0 /\n")
        rock = Rock.from_deck(deck_file, grid=grid)
        pore_volume = rock.porosity * rock.net_to_gross * grid.cell_volumes
        np.testing.assert_allclose(pore_volume, [100, 200, 250, 0])

    def test_porv_alone_is_enough(self):
        deck_file, grid = small_grid("PORV\n 4*500 /\n")
        rock = Rock.from_deck(deck_file, grid=grid)
        np.testing.assert_allclose(rock.porosity * grid.cell_volumes, 500.0)

    def test_without_porv_poro_is_used_unchanged(self):
        deck_file, grid = small_grid("PORO\n 4*0.2 /\n")
        np.testing.assert_allclose(Rock.from_deck(deck_file, grid=grid).porosity, 0.2)

    def test_a_pore_volume_larger_than_the_bulk_volume_is_rejected(self):
        deck_file, grid = small_grid("PORV\n 4*5000 /\n")  # bulk volume is 10 x 10 x 5 = 500
        with pytest.raises(ValidationError, match="porosity above 1"):
            Rock.from_deck(deck_file, grid=grid)

    def test_a_pore_volume_in_a_cell_with_no_net_rock_is_rejected(self):
        deck_file, grid = small_grid("NTG\n 0 3*1 /\nPORV\n 4*100 /\n")
        with pytest.raises(ValidationError, match="no bulk volume"):
            Rock.from_deck(deck_file, grid=grid)

    def test_neither_keyword_is_reported(self):
        deck_file, grid = small_grid("")
        with pytest.raises(ValidationError, match="PORO.*PORV"):
            Rock.from_deck(deck_file, grid=grid)


class TestDimensions:
    def test_spe1_is_within_its_own_limits(self):
        assert check_deck_dimensions(DeckFile(SPE1, encoding="utf-8")) == []

    def test_each_exceeded_limit_is_named(self):
        body = (
            "TABDIMS\n 1 1 /\nEQLDIMS\n 1 /\nWELLDIMS\n 1 /\nVFPPDIMS\n 5 5 5 5 5 0 /\n"
            "SWOF\n 0.2 0 1 0\n 1 1 0 0 /\n 0.2 0 1 0\n 1 1 0 0 /\n"
            "PVTW\n 3000 1 3e-6 .5 0 /\n 3000 1 3e-6 .5 0 /\n"
            "EQUIL\n 1000 3000 1100 0 900 0 /\n 1000 3000 1100 0 900 0 /\n"
            "WELSPECS\n 'A' 'G' 1 1 1* OIL /\n 'B' 'G' 2 2 1* OIL /\n/\n"
        )
        problems = check_deck_dimensions(deck(body))
        text = "\n".join(problems)
        for item in (
            "num_saturation_tables",
            "num_pvt_tables",
            "max_equilibration_regions",
            "max_wells",
        ):
            assert item in text

    def test_a_dimension_keyword_the_deck_omits_is_not_enforced(self):
        assert (
            check_deck_dimensions(
                deck("WELSPECS\n 'A' 'G' 1 1 1* OIL /\n 'B' 'G' 2 2 1* OIL /\n/\n")
            )
            == []
        )

    def test_exceeding_a_limit_warns_and_does_not_fail(self):
        with pytest.warns(DeckDimensionWarning, match="Eclipse would reject"):
            warn_deck_dimensions(
                deck("WELLDIMS\n 1 /\nWELSPECS\n 'A' 'G' 1 1 1* OIL /\n 'B' 'G' 2 2 1* OIL /\n/\n")
            )


def test_a_real_deck_loads_a_case_with_the_deck_derived_run_spec():
    from bores.simulation.case import load_case

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = load_case(DeckFile(SPE1, encoding="utf-8"), temperature=200.0)
    assert case.runspec.phases is not None and case.runspec.phases.is_three_phase
    assert case.runspec.timer is not None
