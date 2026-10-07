import pathlib
import types

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.reservoir.boundary.aquifers.carter_tracy import (
    CarterTracyAquifer,
    load_carter_tracy_aquifer_from_record,
)
from bores.reservoir.boundary.deck import compute_default_aquifer_pressure
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.utils import get_hydrostatic_gradient_factor
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

DATA_DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "data"

AQUCT_RECORD = "1 9035 {pressure} 100 0.25 3.0E-6 5000 100 360 1 1 /"


def build_grid(nx=3, ny=2, nz=2):
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [i * 100.0, j * 100.0, 0.0, i * 100.0, j * 100.0, 5000.0]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        zcorn[2 * k] = 8000.0 + 50.0 * k
        zcorn[2 * k + 1] = 8050.0 + 50.0 * k
    return make_corner_point_grid(coord=coord, zcorn=zcorn, unit_system=UnitSystem.FIELD)


def deck_with_aquct(pressure):
    text = f"AQUCT\n{AQUCT_RECORD.format(pressure=pressure)}\n/\n"
    return DeckFile(text, unit_system=UnitSystem.FIELD)


def fake_pvt(viscosity=0.5):
    static = types.SimpleNamespace(water_reference_viscosity=viscosity)
    return types.SimpleNamespace(region=lambda number: types.SimpleNamespace(static=static))


def test_aquct_parses_a_defaulted_initial_pressure():
    record = deck_with_aquct("1*").get("AQUCT")[0]
    assert record["initial_pressure"] is None
    assert record["datum_depth"] == pytest.approx(9035.0)


def test_aquct_still_parses_an_explicit_initial_pressure():
    record = deck_with_aquct("4800").get("AQUCT")[0]
    assert record["initial_pressure"] == pytest.approx(4800.0)


def test_defaulted_pressure_without_a_source_is_reported():
    deck = deck_with_aquct("1*")
    with pytest.raises(ValidationError, match="defaults its initial pressure"):
        CarterTracyAquifer.from_deck(deck, pvt=fake_pvt())


def test_defaulted_pressure_comes_from_the_callable():
    deck = deck_with_aquct("1*")
    aquifers = CarterTracyAquifer.from_deck(
        deck, pvt=fake_pvt(), get_initial_pressure=lambda record: 5123.0
    )
    assert aquifers[1].initial_pressure == pytest.approx(5123.0)


def test_explicit_pressure_is_not_replaced_by_the_callable():
    deck = deck_with_aquct("4800")
    record = deck.get("AQUCT")[0]
    aquifer = load_carter_tracy_aquifer_from_record(
        record,
        UnitSystem.FIELD,
        pvt=fake_pvt(),
        get_initial_pressure=lambda record: 1.0,
    )
    assert aquifer.initial_pressure == pytest.approx(4800.0)


def test_default_pressure_moves_each_cell_to_the_datum_along_the_water_gradient():
    grid = build_grid()
    pressure = np.full(grid.n_cells, 4000.0)
    positions = np.arange(grid.n_boundary_faces)
    gradient = 0.433

    result = compute_default_aquifer_pressure(
        grid,
        face_positions=positions,
        reservoir_pressure=pressure,
        datum_depth=9000.0,
        water_gradient=gradient,
    )

    boundary_cells = grid.get_boundary_cell_indices()
    expected = np.mean(4000.0 + gradient * (9000.0 - grid.cell_center_depths[boundary_cells]))
    assert result == pytest.approx(expected)


def test_default_pressure_equals_cell_pressure_at_the_cells_own_depth():
    grid = build_grid()
    boundary_cells = grid.get_boundary_cell_indices()
    cell = int(boundary_cells[0])
    pressure = np.full(grid.n_cells, 4000.0)
    face_position = int(
        np.flatnonzero(grid.face_cell_indices[grid.boundary_face_indices, 0] == cell)[0]
    )
    result = compute_default_aquifer_pressure(
        grid,
        face_positions=np.array([face_position]),
        reservoir_pressure=pressure,
        datum_depth=float(grid.cell_center_depths[cell]),
        water_gradient=0.433,
    )
    assert result == pytest.approx(4000.0)


def test_default_pressure_with_no_attached_cell_is_rejected():
    grid = build_grid()
    with pytest.raises(ValidationError, match="not attached"):
        compute_default_aquifer_pressure(
            grid,
            face_positions=np.array([], dtype=np.int32),
            reservoir_pressure=np.zeros(grid.n_cells),
            datum_depth=9000.0,
            water_gradient=0.433,
        )


def load_spe1_with_aquifer(pressure):
    text = (DATA_DIRECTORY / "SPE1CASE1.DATA").read_text()
    aquifer = (
        f"\nAQUCT\n{AQUCT_RECORD.format(pressure=pressure)}\n/\n\n"
        "AQUANCON\n 1 10 10 1 10 1 3 I+ 1* 1.0 NO /\n/\n\n"
    )
    assert "\nSUMMARY" in text
    deck = DeckFile(
        text.replace("\nSUMMARY", aquifer + "\nSUMMARY", 1), unit_system=UnitSystem.FIELD
    )
    return SimulationCase.from_deck(
        deck,
        default_wellbore=homogeneous_wellbore(
            tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD
        ),
        temperature=200.0,
    )


def test_case_loads_with_an_explicit_aquifer_pressure():
    case = load_spe1_with_aquifer("4800")
    initial_pressures = case.model.boundary_conditions.aquifers.initial_pressures
    assert initial_pressures[0] == pytest.approx(4800.0)


def test_case_loads_with_a_defaulted_aquifer_pressure():
    case = load_spe1_with_aquifer("1*")
    model = case.model
    grid = model.reservoir.grid
    static = model.fluid.pvt.region(1).static
    water_density = static.stock_tank_water_density / static.water_reference_fvf
    gradient = water_density * get_hydrostatic_gradient_factor(UnitSystem.FIELD)

    nx, ny, nz = 10, 10, 3
    cells = np.array([nx - 1 + nx * (j + ny * k) for k in range(nz) for j in range(ny)])
    expected = np.mean(
        case.initial_state.pressure[cells] + gradient * (9035.0 - grid.cell_center_depths[cells])
    )

    initial_pressures = model.boundary_conditions.aquifers.initial_pressures
    assert initial_pressures[0] == pytest.approx(expected, rel=1e-5)
    assert 4800.0 < initial_pressures[0] < 5500.0
