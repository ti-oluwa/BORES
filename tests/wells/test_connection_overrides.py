"""`COMPDAT` kh and pressure equivalent radius, and `WELSPECS` drainage radius, in the well index."""

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.types import UnitSystem
from bores.units import get_conversion_factors
from bores.wells.base import Perforation, convert_perforation
from bores.wells.deck import load_wells, positive_or_none
from bores.wells.indices.wells import (
    compute_equivalent_radius_well_index,
    compute_peaceman_well_index,
    resolve_connection_terms,
)

DECK = (
    "FIELD\nSPECGRID\n 3 3 2 1 F /\nDX\n 18*100 /\nDY\n 18*100 /\nDZ\n 18*20 /\n"
    "TOPS\n 9*8000 /\nPERMX\n 18*100 /\nPERMY\n 18*100 /\nPERMZ\n 18*10 /\nPORO\n 18*0.2 /\n"
)


def grid() -> Grid:
    return Grid.from_deck(DeckFile(DECK))


def terms(perforation: Perforation, **kwargs):
    return resolve_connection_terms(
        perforation=perforation,
        grid=grid(),
        cell_index=4,
        partial_penetration_fraction=1.0,
        wellbore_radius=0.25,
        kx=100.0,
        ky=100.0,
        kz=10.0,
        **kwargs,
    )


def vertical(**fields) -> Perforation:
    return Perforation(
        top_depth=8000.0, bottom_depth=8020.0, wellbore_radius=0.25, skin=0.5, **fields
    )


class TestPeacemanBranch:
    def test_with_no_override_nothing_changes(self):
        baseline, _ = terms(vertical())
        assert baseline > 0

    def test_kh_replaces_permeability_times_thickness_and_is_not_scaled_again(self):
        _, conductivity = terms(vertical(permeability_thickness=3000.0), net_to_gross=0.5)
        assert conductivity == pytest.approx(3000.0)  # no NTG, no partial penetration

        baseline, baseline_conductivity = terms(vertical(), net_to_gross=0.5)
        overridden, _ = terms(vertical(permeability_thickness=3000.0), net_to_gross=0.5)
        assert overridden / baseline == pytest.approx(3000.0 / baseline_conductivity)

    def test_the_equivalent_radius_replaces_the_one_worked_out(self):
        expected = compute_peaceman_well_index(
            permeability=100.0,
            interval_thickness=20.0,
            wellbore_radius=0.25,
            effective_drainage_radius=15.0,
            skin_factor=0.5,
            net_to_gross=1.0,
            regime_constant=-3 / 4,
        )
        index, _ = terms(vertical(pressure_equivalent_radius=15.0))
        assert index == pytest.approx(expected)

    def test_a_larger_equivalent_radius_gives_a_smaller_index(self):
        small, _ = terms(vertical(pressure_equivalent_radius=5.0))
        large, _ = terms(vertical(pressure_equivalent_radius=50.0))
        assert large < small

    def test_both_together(self):
        expected = compute_peaceman_well_index(
            permeability=2000.0,
            interval_thickness=1.0,
            wellbore_radius=0.25,
            effective_drainage_radius=15.0,
            skin_factor=0.5,
            net_to_gross=1.0,
            regime_constant=-3 / 4,
        )
        index, conductivity = terms(
            vertical(permeability_thickness=2000.0, pressure_equivalent_radius=15.0)
        )
        assert index == pytest.approx(expected) and conductivity == pytest.approx(2000.0)

    def test_an_explicit_connection_factor_still_wins_over_both(self):
        index, conductivity = terms(
            vertical(
                connection_factor_override=7.0,
                permeability_thickness=2000.0,
                pressure_equivalent_radius=15.0,
            )
        )
        assert index == 7.0 and conductivity is None

    def test_the_wpimult_multiplier_still_scales_an_overridden_index(self):
        plain, _ = terms(vertical(permeability_thickness=2000.0))
        scaled, _ = terms(
            vertical(permeability_thickness=2000.0, connection_factor_multiplier=3.0)
        )
        assert scaled == pytest.approx(3.0 * plain)


class TestEquivalentRadiusBranch:
    BASE = dict(
        permeability=100.0,
        cell_volume=200000.0,
        completion_length=20.0,
        wellbore_radius=0.25,
        skin=0.5,
    )

    def test_zero_means_work_it_out(self):
        default = compute_equivalent_radius_well_index(**self.BASE)
        explicit_zero = compute_equivalent_radius_well_index(
            **self.BASE, equivalent_radius=0.0, permeability_thickness=0.0
        )
        assert default == explicit_zero

    def test_the_overrides_follow_the_formula(self):
        index = compute_equivalent_radius_well_index(
            **self.BASE, equivalent_radius=12.0, permeability_thickness=1500.0
        )
        assert index == pytest.approx(1500.0 / (np.log(12.0 / 0.25) + 0.5))


class TestValidationAndUnits:
    def test_non_positive_values_are_rejected(self):
        with pytest.raises(ValidationError, match="permeability_thickness"):
            vertical(permeability_thickness=0.0)
        with pytest.raises(ValidationError, match="pressure_equivalent_radius"):
            vertical(pressure_equivalent_radius=-1.0)

    def test_conversion_scales_kh_by_permeability_times_length_and_the_radius_by_length(self):
        factors = get_conversion_factors(UnitSystem.FIELD, UnitSystem.METRIC)
        length = factors["length"]
        connection = factors["permeability"] * length
        converted = convert_perforation(
            vertical(permeability_thickness=2000.0, pressure_equivalent_radius=10.0),
            length_factor=length,
            connection_factor=connection,
        )
        assert converted.permeability_thickness == pytest.approx(2000.0 * connection)
        assert converted.pressure_equivalent_radius == pytest.approx(10.0 * length)

    def test_a_deck_value_of_nothing_zero_or_negative_means_work_it_out(self):
        assert positive_or_none(None) is None and positive_or_none(0.0) is None
        assert positive_or_none(-1.0) is None and positive_or_none(2.5) == 2.5


class TestFromTheDeck:
    WELLS = (
        "WELSPECS\n 'P' 'G' 1 1 1* OIL 120 /\n 'Q' 'G' 2 2 1* OIL /\n/\n"
        "COMPDAT\n 'P' 1 1 1 1 OPEN 1* 1* 0.5 250 2.5 1* Z 0.9 /\n 'Q' 2 2 1 1 OPEN 1* 1* 0.5 /\n/\n"
    )

    def test_compdat_and_welspecs_values_reach_the_well(self):
        deck = DeckFile(DECK + self.WELLS)
        wells = load_wells(deck, Grid.from_deck(deck))
        override, plain = wells["P"], wells["Q"]
        assert override.drainage_radius == 120.0
        assert override.perforations[0].permeability_thickness == 250.0
        assert override.perforations[0].pressure_equivalent_radius == 0.9
        assert plain.drainage_radius is None
        assert plain.perforations[0].permeability_thickness is None
        assert plain.perforations[0].pressure_equivalent_radius is None

    def test_a_well_converts_its_drainage_radius(self):
        deck = DeckFile(DECK + self.WELLS)
        well = load_wells(deck, Grid.from_deck(deck))["P"]
        length = get_conversion_factors(UnitSystem.FIELD, UnitSystem.METRIC)["length"]
        assert well.convert(UnitSystem.METRIC).drainage_radius == pytest.approx(120.0 * length)

    def test_a_non_positive_drainage_radius_on_a_well_is_rejected(self):
        from bores.wells.factories import make_producer

        with pytest.raises(ValidationError, match="drainage_radius"):
            make_producer(
                name="P",
                perforation_depths=(8000.0, 8020.0),
                surface_location=(0.0, 0.0),
                target_bhp=1000.0,
                drainage_radius=-5.0,
            )
