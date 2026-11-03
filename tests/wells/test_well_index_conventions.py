"""Both well index branches share one convention, and the regime constant is configurable."""

import numpy as np
import pytest

from bores.constants import Constants, ConstantsContext, c
from bores.deck import DeckFile
from bores.grids.base import Grid
from bores.simulation.deck import load_run_spec
from bores.simulation.options import PhysicsOptions
from bores.simulation.spec import RunSpec
from bores.types import UnitSystem
from bores.wells.base import Perforation
from bores.wells.indices.wells import (
    compute_equivalent_radius_well_index,
    compute_peaceman_well_index,
    resolve_connection_terms,
)

DARCY_FT3_PER_DAY = 6.3283e-3  # mD*ft/cP*psi to ft3/day, no 2*pi
DECK = (
    "FIELD\nSPECGRID\n 3 3 2 1 F /\nDX\n 18*100 /\nDY\n 18*100 /\nDZ\n 18*20 /\n"
    "TOPS\n 9*8000 /\nPERMX\n 18*100 /\nPERMY\n 18*100 /\nPERMZ\n 18*10 /\nPORO\n 18*0.2 /\n"
)
PERMEABILITY, LENGTH, RADIUS, SKIN, EQUIVALENT_RADIUS = 100.0, 20.0, 0.25, 0.5, 15.0


def terms(**kwargs):
    perforation = Perforation(
        top_depth=8000.0,
        bottom_depth=8020.0,
        wellbore_radius=RADIUS,
        skin=SKIN,
        pressure_equivalent_radius=EQUIVALENT_RADIUS,
    )
    return resolve_connection_terms(
        perforation=perforation,
        grid=Grid.from_deck(DeckFile(DECK)),
        cell_index=4,
        partial_penetration_fraction=1.0,
        wellbore_radius=RADIUS,
        kx=PERMEABILITY,
        ky=PERMEABILITY,
        kz=10.0,
        **kwargs,
    )


class TestOneConvention:
    def test_the_two_branches_agree_when_given_the_same_radius_and_no_regime_constant(self):
        peaceman = compute_peaceman_well_index(
            permeability=PERMEABILITY,
            interval_thickness=LENGTH,
            wellbore_radius=RADIUS,
            effective_drainage_radius=EQUIVALENT_RADIUS,
            skin_factor=SKIN,
            net_to_gross=1.0,
            regime_constant=0.0,
        )
        equivalent = compute_equivalent_radius_well_index(
            permeability=PERMEABILITY,
            cell_volume=1.0,
            completion_length=LENGTH,
            wellbore_radius=RADIUS,
            skin=SKIN,
            equivalent_radius=EQUIVALENT_RADIUS,
        )
        assert equivalent == pytest.approx(peaceman)

    def test_the_radial_flow_constant_turns_either_index_into_the_textbook_rate(self):
        textbook = (
            2
            * np.pi
            * DARCY_FT3_PER_DAY
            * PERMEABILITY
            * LENGTH
            / (np.log(EQUIVALENT_RADIUS / RADIUS) + SKIN)
        )
        equivalent = compute_equivalent_radius_well_index(
            permeability=PERMEABILITY,
            cell_volume=1.0,
            completion_length=LENGTH,
            wellbore_radius=RADIUS,
            skin=SKIN,
            equivalent_radius=EQUIVALENT_RADIUS,
        )
        peaceman = compute_peaceman_well_index(
            permeability=PERMEABILITY,
            interval_thickness=LENGTH,
            wellbore_radius=RADIUS,
            effective_drainage_radius=EQUIVALENT_RADIUS,
            skin_factor=SKIN,
            net_to_gross=1.0,
            regime_constant=0.0,
        )
        assert c.RADIAL_FLOW_CONSTANT_FIELD * equivalent == pytest.approx(textbook, rel=1e-3)
        assert c.RADIAL_FLOW_CONSTANT_FIELD * peaceman == pytest.approx(textbook, rel=1e-3)

    def test_the_index_is_the_conductivity_over_the_resistance_plus_skin(self):
        equivalent = compute_equivalent_radius_well_index(
            permeability=PERMEABILITY,
            cell_volume=1.0,
            completion_length=LENGTH,
            wellbore_radius=RADIUS,
            skin=SKIN,
            equivalent_radius=EQUIVALENT_RADIUS,
        )
        assert equivalent * (np.log(EQUIVALENT_RADIUS / RADIUS) + SKIN) == pytest.approx(
            PERMEABILITY * LENGTH
        )


class TestRegimeConstant:
    def test_the_constant_is_registered_as_pseudo_steady_state(self):
        assert c.WELL_INDEX_REGIME_CONSTANT == -0.75

    def test_none_uses_the_constant_and_matches_an_explicit_value(self):
        default, _ = terms()
        explicit, _ = terms(regime_constant=-0.75)
        assert default == explicit

    def test_a_given_value_changes_the_index_by_the_formula(self):
        zero, conductivity = terms(regime_constant=0.0)
        assert zero == pytest.approx(conductivity / (np.log(EQUIVALENT_RADIUS / RADIUS) + SKIN))
        transient, _ = terms(regime_constant=0.5)
        assert transient < zero < terms()[0]

    def test_the_default_follows_the_runtime_constants(self):
        with ConstantsContext(Constants({"WELL_INDEX_REGIME_CONSTANT": 0.0})):
            overridden, _ = terms()
        assert overridden == pytest.approx(terms(regime_constant=0.0)[0])

    def test_the_run_spec_defaults_to_the_constant_and_a_deck_loads_zero(self):
        assert RunSpec().physics_options.well_index_regime_constant is None
        spec = load_run_spec(DeckFile(DECK + "TSTEP\n 5 /\n"))
        assert spec.physics_options.well_index_regime_constant == 0.0

    def test_a_user_can_set_it_on_the_run_spec_options(self):
        spec = RunSpec(
            unit_system=UnitSystem.FIELD,
            physics_options=PhysicsOptions(well_index_regime_constant=0.5),
        )
        assert spec.physics_options.well_index_regime_constant == 0.5


class TestNetToGross:
    def test_the_equivalent_radius_function_scales_a_worked_out_thickness_only(self):
        base = dict(
            permeability=PERMEABILITY,
            cell_volume=1.0,
            completion_length=LENGTH,
            wellbore_radius=RADIUS,
            skin=SKIN,
            equivalent_radius=EQUIVALENT_RADIUS,
        )
        full = compute_equivalent_radius_well_index(**base)
        assert compute_equivalent_radius_well_index(**base, net_to_gross=0.5) == pytest.approx(
            0.5 * full
        )
        given = compute_equivalent_radius_well_index(**base, permeability_thickness=1500.0)
        assert compute_equivalent_radius_well_index(
            **base, permeability_thickness=1500.0, net_to_gross=0.5
        ) == pytest.approx(given)

    @pytest.fixture
    def non_cartesian(self, monkeypatch):
        monkeypatch.setattr(
            "bores.wells.indices.wells.is_locally_cartesian", lambda grid, cell_index: False
        )

    def test_both_branches_apply_it_to_the_index_and_the_conductivity(self, monkeypatch):
        cartesian_full, cartesian_conductivity = terms()
        cartesian_half, half_conductivity = terms(net_to_gross=0.5)
        assert cartesian_half == pytest.approx(0.5 * cartesian_full)
        assert half_conductivity == pytest.approx(0.5 * cartesian_conductivity)

        monkeypatch.setattr(
            "bores.wells.indices.wells.is_locally_cartesian", lambda grid, cell_index: False
        )
        full, conductivity = terms()
        half, scaled_conductivity = terms(net_to_gross=0.5)
        assert half == pytest.approx(0.5 * full)
        assert scaled_conductivity == pytest.approx(0.5 * conductivity)
        # The index is still the conductivity over the resistance plus skin
        assert half == pytest.approx(
            scaled_conductivity / (np.log(EQUIVALENT_RADIUS / RADIUS) + SKIN)
        )

    def test_a_given_kh_is_not_scaled_by_it(self, non_cartesian):
        perforation = Perforation(
            top_depth=8000.0,
            bottom_depth=8020.0,
            wellbore_radius=RADIUS,
            skin=SKIN,
            permeability_thickness=1500.0,
        )

        def with_ntg(net_to_gross: float):
            return resolve_connection_terms(
                perforation=perforation,
                grid=Grid.from_deck(DeckFile(DECK)),
                cell_index=4,
                partial_penetration_fraction=1.0,
                wellbore_radius=RADIUS,
                kx=PERMEABILITY,
                ky=PERMEABILITY,
                kz=10.0,
                net_to_gross=net_to_gross,
            )

        assert with_ntg(0.5) == with_ntg(1.0)

    def test_the_compiled_path_applies_a_scalar_or_a_per_cell_net_to_gross(self):
        from bores.types import Orientation
        from bores.wells.compile import compile_perforations
        from bores.wells.factories import make_producer, make_wells

        grid = Grid.from_deck(DeckFile(DECK))
        well, _ = make_producer(
            name="P", perforation_depths=(8000.0, 8020.0), surface_location=(150.0, 150.0), target_bhp=1000.0
        )
        permeabilities = {
            Orientation.X: np.full(grid.n_cells, 100.0),
            Orientation.Y: np.full(grid.n_cells, 100.0),
            Orientation.Z: np.full(grid.n_cells, 10.0),
        }

        def first_index(**kwargs) -> float:
            compiled = compile_perforations(
                names=["P"], wells=make_wells(well), grid=grid, permeabilities=permeabilities, **kwargs
            )
            return float(compiled.well_indices[0])

        full = first_index()
        assert first_index(net_to_gross=0.5) == pytest.approx(0.5 * full)
        assert first_index(net_to_gross=np.full(grid.n_cells, 0.5)) == pytest.approx(0.5 * full)
