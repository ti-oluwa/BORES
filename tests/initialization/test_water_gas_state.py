import warnings
from pathlib import Path

import numpy as np
import pytest

from bores.blackoil.pvt.factories import build_pvt_dataset
from bores.blackoil.pvt.tables import PVTTables
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.initialization import initialize_reservoir_state, initialize_water_gas_state
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem

SPE1_DECK = Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA"
SALINITIES = np.array([10000.0, 30000.0, 60000.0])
PRESSURES = np.array([1000.0, 2500.0, 4000.0, 5500.0])
TEMPERATURE = np.full(4, 200.0)
SALINITY = np.full(4, 30000.0)
REGION = np.ones(4, dtype=int)


class SingleRegionPVT:
    def __init__(self, tables):
        self.region_tables = type("Region", (), {"tables": tables})()

    def region(self, pvtnum):
        return self.region_tables


@pytest.fixture(scope="module")
def tables():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = build_pvt_dataset(
            pressures=np.linspace(500.0, 6000.0, 24),
            temperatures=np.array([150.0, 200.0, 250.0]),
            gas_gravity=0.7,
            build_oil_data=False,
            salinities=SALINITIES,
        )
        return PVTTables.from_dataset(dataset, interpolation_method="linear")


def saturated_rsw(tables, pressure):
    return tables.gas.solubility_in_water(pressure, TEMPERATURE, salinity=SALINITY)


def run(tables, **kwargs):
    return initialize_water_gas_state(
        pvt=SingleRegionPVT(tables),
        pvt_region_index=REGION,
        pressure=PRESSURES,
        temperature=TEMPERATURE,
        salinity=SALINITY,
        **kwargs,
    )


def test_default_is_saturated_water(tables):
    rsw, bubble_point = run(tables)
    assert np.allclose(rsw, saturated_rsw(tables, PRESSURES), rtol=1e-6)
    assert np.allclose(bubble_point, PRESSURES, rtol=1e-4)


def test_explicit_undersaturated_rsw_gives_the_pressure_that_holds_it(tables):
    explicit = np.asarray(saturated_rsw(tables, 0.6 * PRESSURES))
    rsw, bubble_point = run(tables, gas_solubility_in_water=explicit)
    assert np.allclose(rsw, explicit, rtol=1e-6)
    assert np.allclose(bubble_point, 0.6 * PRESSURES, rtol=1e-3)


def test_explicit_rsw_above_saturation_is_capped_at_the_pressure(tables):
    explicit = 1.5 * np.asarray(saturated_rsw(tables, PRESSURES))
    _, bubble_point = run(tables, gas_solubility_in_water=explicit)
    assert np.allclose(bubble_point, PRESSURES)


def test_region_without_rsw_data(tables):
    no_gas = type("Tables", (), {"gas": None, "water": None})()
    pvt = SingleRegionPVT(no_gas)
    kwargs = dict(pvt=pvt, pvt_region_index=REGION, pressure=PRESSURES, temperature=TEMPERATURE)
    rsw, bubble_point = initialize_water_gas_state(**kwargs)
    assert not rsw.any() and not bubble_point.any()
    explicit = np.full(4, 3.0)
    rsw, bubble_point = initialize_water_gas_state(gas_solubility_in_water=explicit, **kwargs)
    assert np.allclose(rsw, 3.0) and np.allclose(bubble_point, PRESSURES)


@pytest.fixture(scope="module")
def spe1():
    deck_file = DeckFile(str(SPE1_DECK), encoding="utf-8", unit_system=UnitSystem.FIELD)
    case = SimulationCase.from_deck(deck_file, temperature=200.0)
    return deck_file, case


def initialize(spe1, **kwargs):
    deck_file, case = spe1
    return initialize_reservoir_state(
        case.model.reservoir,
        case.model.fluid.pvt,
        deck_file=deck_file,
        satfunc=None,
        temperature=200.0,
        **kwargs,
    )


def test_deck_without_rsw_data_keeps_zero_dissolved_gas_in_water(spe1):
    state = initialize(spe1)
    assert not state.gas_solubility_in_water.any()
    assert not state.water_bubble_point_pressure.any()
    assert not state.dissolved_gas_mass_in_water.any()


def test_explicit_rsw_reaches_the_state(spe1):
    n_cells = spe1[1].model.reservoir.n_cells
    state = initialize(spe1, gas_solubility_in_water=np.full(n_cells, 4.0))
    assert np.allclose(state.gas_solubility_in_water, 4.0)
    assert np.allclose(state.water_bubble_point_pressure, state.pressure)
    assert (state.dissolved_gas_mass_in_water > 0).all()


def test_rsw_keyword_in_the_deck_reaches_the_state(spe1, tmp_path):
    n_cells = spe1[1].model.reservoir.n_cells
    lines = SPE1_DECK.read_text().split("\n")
    at = next(i for i, line in enumerate(lines) if line.strip() == "SOLUTION")
    lines[at + 1 : at + 1] = ["RSW", f"{n_cells}*5.0 /", ""]
    path = tmp_path / "spe1_rsw.DATA"
    path.write_text("\n".join(lines))
    deck_file = DeckFile(str(path), encoding="utf-8", unit_system=UnitSystem.FIELD)
    case = SimulationCase.from_deck(deck_file, temperature=200.0)
    state = initialize_reservoir_state(
        case.model.reservoir,
        case.model.fluid.pvt,
        deck_file=deck_file,
        satfunc=None,
        temperature=200.0,
    )
    assert np.allclose(state.gas_solubility_in_water, 5.0)


@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test_invalid_explicit_rsw_is_rejected(spe1, bad):
    values = np.full(spe1[1].model.reservoir.n_cells, 4.0)
    values[0] = bad
    with pytest.raises(ValidationError):
        initialize(spe1, gas_solubility_in_water=values)


def test_wrong_shape_is_rejected(spe1):
    with pytest.raises(ValidationError):
        initialize(spe1, gas_solubility_in_water=np.zeros(3))


def test_more_bisection_iterations_tighten_the_bubble_point(tables):
    explicit = np.asarray(saturated_rsw(tables, 0.6 * PRESSURES))
    _, coarse = run(tables, gas_solubility_in_water=explicit, iterations=3)
    _, fine = run(tables, gas_solubility_in_water=explicit)
    assert np.abs(coarse - 0.6 * PRESSURES).max() > np.abs(fine - 0.6 * PRESSURES).max()
