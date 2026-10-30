"""Tests for loading the imbibition end-point arrays from a deck."""

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.reservoir.rock import IMBIBITION_ENDPOINT_KEYWORDS, ImbibitionEndpoints
from bores.types import UnitSystem


def _deck(body: str, units: str = "FIELD") -> DeckFile:
    return DeckFile(
        f"{units}\nRUNSPEC\nDIMENS\n 3 1 1 /\nSATOPTS\n HYSTER /\nENDSCALE\n/\nPROPS\n{body}"
    )


def test_a_deck_without_imbibition_endpoints_gives_none():
    assert ImbibitionEndpoints.from_deck(_deck("SWL\n 0.2 0.2 0.2 /\n"), n_cells=3) is None


def test_each_keyword_lands_on_its_own_field_in_the_decks_units():
    deck_file = _deck(
        "ISWL\n 0.18 0.2 0.22 /\n"
        "ISGCR\n 0.05 0.06 0.07 /\n"
        "IKRW\n 0.4 0.5 0.6 /\n"
        "IPCW\n 10 20 30 /\n"
    )
    endpoints = ImbibitionEndpoints.from_deck(deck_file, n_cells=3, dtype=np.float64)
    assert endpoints is not None
    np.testing.assert_allclose(endpoints.connate_water_saturation, [0.18, 0.2, 0.22])
    np.testing.assert_allclose(endpoints.critical_gas_saturation, [0.05, 0.06, 0.07])
    np.testing.assert_allclose(endpoints.maximum_water_relative_permeability, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(endpoints.maximum_oil_water_capillary_pressure, [10, 20, 30])
    assert endpoints.maximum_gas_oil_capillary_pressure is None
    assert endpoints.critical_water_saturation is None
    assert endpoints.unit_system == UnitSystem.FIELD


def test_every_keyword_this_class_reads_is_a_distinct_one():
    assert len(set(IMBIBITION_ENDPOINT_KEYWORDS)) == len(IMBIBITION_ENDPOINT_KEYWORDS) == 19


def test_only_the_capillary_pressures_change_with_the_unit_system():
    endpoints = ImbibitionEndpoints(
        critical_gas_saturation=np.array([0.05, np.nan]),
        maximum_oil_water_capillary_pressure=np.array([10.0, np.nan]),
        maximum_gas_oil_capillary_pressure=np.array([4.0, 8.0]),
        unit_system=UnitSystem.FIELD,
    )
    metric = endpoints.convert(UnitSystem.METRIC)
    assert metric.unit_system == UnitSystem.METRIC
    np.testing.assert_allclose(
        metric.maximum_oil_water_capillary_pressure, [10.0 * 0.0689475729, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        metric.maximum_gas_oil_capillary_pressure, [0.27579, 0.55158], rtol=1e-4
    )
    np.testing.assert_array_equal(
        metric.critical_gas_saturation, endpoints.critical_gas_saturation
    )
    assert metric.convert(UnitSystem.FIELD).maximum_gas_oil_capillary_pressure == pytest.approx(
        endpoints.maximum_gas_oil_capillary_pressure
    )


def test_unset_cells_stay_nan_so_they_take_the_table_endpoint():
    endpoints = ImbibitionEndpoints.from_deck(
        _deck("EQUALS\nISGCR 0.1 1 2 1 1 1 1 /\n/\n"), n_cells=3, dtype=np.float64
    )
    assert endpoints is not None
    assert np.isnan(endpoints.critical_gas_saturation[2])


@pytest.mark.parametrize(
    "body",
    ["ISWL\n 0.2 1.4 0.2 /\n", "IKRG\n -0.1 0.2 0.2 /\n", "IPCG\n -5 1 1 /\n"],
)
def test_out_of_range_values_are_rejected(body):
    with pytest.raises(ValidationError):
        ImbibitionEndpoints.from_deck(_deck(body), n_cells=3)


def test_a_wrong_number_of_values_is_rejected():
    with pytest.raises(ValidationError):
        ImbibitionEndpoints.from_deck(_deck("ISWL\n 0.2 0.2 /\n"), n_cells=3)
