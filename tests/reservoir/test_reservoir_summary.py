"""Tests for the field, region and block reservoir summary vectors and the time stepping ones."""

import types

import numpy as np
import pytest

from bores.constants import c
from bores.errors import SummaryError
from bores.reservoir import summary as rs
from bores.schedule.base import ScheduleContext
from bores.timestep import summary as ts
from bores.timestep.control import StepMetrics
from bores.types import UnitSystem

PORE_VOLUMES = np.array([1000.0, 2000.0, 3000.0, 4000.0])


def make_workspace(pressure=3000.0):
    state = types.SimpleNamespace(
        pressure=np.full(4, pressure),
        oil_saturation=np.array([0.5, 0.6, 0.4, 0.0]),
        water_saturation=np.array([0.3, 0.3, 0.3, 0.3]),
        gas_saturation=np.array([0.2, 0.1, 0.3, 0.7]),
        solution_gor=np.array([500.0, 400.0, 300.0, 200.0]),
        gas_solubility_in_water=np.array([20.0, 20.0, 10.0, np.nan]),
    )
    pvt = types.SimpleNamespace(
        oil_formation_volume_factor=np.array([1.2, 1.25, 1.3, 1.35]),
        water_formation_volume_factor=np.ones(4),
        gas_formation_volume_factor=np.array([0.005, 0.005, 0.006, 0.006]),
        vaporized_oil_gas_ratio=np.array([1e-4, 2e-4, np.nan, 5e-5]),
    )
    return types.SimpleNamespace(reservoir=state, physics=types.SimpleNamespace(pvt=pvt))


def make_model(regions=None, unit_system=UnitSystem.FIELD):
    reservoir = types.SimpleNamespace(
        compute_pore_volumes=lambda pressure: PORE_VOLUMES * (1.0 + 1e-5 * (pressure - 3000.0)),
        regions=None
        if regions is None
        else types.SimpleNamespace(fluid_in_place_region=np.asarray(regions)),
        grid=types.SimpleNamespace(
            dimensions=types.SimpleNamespace(
                nx=2, ny=2, nz=1, flat_index=lambda i, j, k: i + 2 * j + 4 * k
            )
        ),
    )
    return types.SimpleNamespace(reservoir=reservoir, unit_system=unit_system)


@pytest.fixture
def workspace(monkeypatch):
    workspace = make_workspace()
    monkeypatch.setattr(rs, "get_workspace", lambda *, context: workspace)
    return workspace


def cell_volumes():
    to_stb = c.CUBIC_FEET_TO_STB
    oil = np.array([0.5, 0.6, 0.4, 0.0]) * PORE_VOLUMES / np.array([1.2, 1.25, 1.3, 1.35])
    water = 0.3 * PORE_VOLUMES
    free_gas = (
        np.array([0.2, 0.1, 0.3, 0.7]) * PORE_VOLUMES / np.array([0.005, 0.005, 0.006, 0.006])
    )
    vaporized = np.array([1e-4, 2e-4, 0.0, 5e-5]) * free_gas / to_stb
    gas_in_oil = np.array([500.0, 400.0, 300.0, 200.0]) * oil * to_stb
    gas_in_water = np.array([20.0, 20.0, 10.0, 0.0]) * water * to_stb
    return oil, vaporized, free_gas, gas_in_oil, gas_in_water, water


def test_field_in_place_splits_add_up_to_the_totals(workspace):
    model = make_model()
    oil, vaporized, free_gas, gas_in_oil, gas_in_water, water = cell_volumes()

    assert rs.FieldOilInPlaceInOil()(model, None) == pytest.approx(oil.sum())
    assert rs.FieldOilInPlaceInGas()(model, None) == pytest.approx(vaporized.sum())
    assert rs.FieldOilInPlace()(model, None) == pytest.approx(oil.sum() + vaporized.sum())
    assert rs.FieldGasInPlaceFree()(model, None) == pytest.approx(free_gas.sum())
    assert rs.FieldGasInPlaceDissolved()(model, None) == pytest.approx(
        gas_in_oil.sum() + gas_in_water.sum()
    )
    assert rs.FieldGasInPlace()(model, None) == pytest.approx(
        free_gas.sum() + gas_in_oil.sum() + gas_in_water.sum()
    )
    assert rs.FieldWaterInPlace()(model, None) == pytest.approx(water.sum())


def test_field_equals_the_sum_of_its_regions(workspace):
    model = make_model(regions=[1, 1, 2, 2])
    for field_cls, region_cls in [
        (rs.FieldOilInPlace, rs.RegionOilInPlace),
        (rs.FieldGasInPlace, rs.RegionGasInPlace),
        (rs.FieldGasInPlaceDissolved, rs.RegionGasInPlaceDissolved),
        (rs.FieldPoreVolume, rs.RegionPoreVolume),
    ]:
        regions_total = sum(region_cls(region=r)(model, None) for r in (1, 2))
        assert field_cls()(model, None) == pytest.approx(regions_total)


def test_pore_volume_follows_the_current_pressure(workspace):
    model = make_model()
    assert rs.FieldPoreVolume()(model, None) == pytest.approx(PORE_VOLUMES.sum())
    workspace.reservoir.pressure = np.full(4, 4000.0)
    assert rs.FieldPoreVolume()(model, None) == pytest.approx(PORE_VOLUMES.sum() * 1.01)


def test_average_pressure_weights(workspace):
    workspace.reservoir.pressure = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    model = make_model(regions=[1, 1, 2, 2])
    pressures = workspace.reservoir.pressure

    # Pore volumes shift with pressure here, so weight by the model's own current volumes.
    volumes = model.reservoir.compute_pore_volumes(pressures)
    assert rs.FieldAveragePressure()(model, None) == pytest.approx(
        (pressures * volumes).sum() / volumes.sum()
    )

    hydrocarbon = (np.array([0.5, 0.6, 0.4, 0.0]) + np.array([0.2, 0.1, 0.3, 0.7])) * volumes
    assert rs.FieldHydrocarbonAveragePressure()(model, None) == pytest.approx(
        (pressures * hydrocarbon).sum() / hydrocarbon.sum()
    )
    assert rs.RegionAveragePressure(region=2)(model, None) == pytest.approx(
        (pressures[2:] * volumes[2:]).sum() / volumes[2:].sum()
    )


def test_hydrocarbon_average_pressure_is_nan_without_hydrocarbon(workspace):
    workspace.reservoir.oil_saturation = np.zeros(4)
    workspace.reservoir.gas_saturation = np.zeros(4)
    assert np.isnan(rs.FieldHydrocarbonAveragePressure()(make_model(), None))


def test_region_with_no_cells_is_an_error(workspace):
    with pytest.raises(SummaryError, match="region 3 has no cells"):
        rs.RegionOilInPlace(region=3)(make_model(regions=[1, 1, 2, 2]), None)


def test_no_regions_means_region_one_covers_the_field(workspace):
    model = make_model()
    assert rs.RegionOilInPlace(region=1)(model, None) == pytest.approx(
        rs.FieldOilInPlace()(model, None)
    )
    with pytest.raises(SummaryError):
        rs.RegionOilInPlace(region=2)(model, None)


def test_report_keys():
    assert rs.FieldOilInPlace().key == "FOIP"
    assert rs.RegionGasInPlaceFree(region=4).key == "RGIPG:4"
    assert rs.BlockPoreVolume(i=0, j=1, k=0).key == "BPORV:1,2,1"


def test_block_pore_volume_reads_the_model(workspace):
    model = make_model()
    assert rs.BlockPoreVolume(i=1, j=1, k=0)(model, None) == pytest.approx(4000.0)
    workspace.reservoir.pressure = np.full(4, 4000.0)
    assert rs.BlockPoreVolume(i=0, j=1, k=0)(model, None) == pytest.approx(3000.0 * 1.01)


def timer_context(metrics):
    from bores.timestep.control import Timer

    timer = Timer(
        initial_step_size=1.0, maximum_step_size=10.0, minimum_step_size=0.1, simulation_time=100.0
    )
    timer.recent_metrics.extend(metrics)
    return ScheduleContext(time=1.0, extra={"timer": timer})


def test_timestep_vectors_read_the_latest_accepted_step():
    context = timer_context([
        StepMetrics(step=1, step_size=2.0, newton_iterations=4, success=True),
        StepMetrics(step=2, step_size=8.0, newton_iterations=12, success=False),
    ])
    assert ts.TimeStepSize()(None, context) == pytest.approx(2.0)
    assert ts.NewtonIterations()(None, context) == 4


def test_timestep_vectors_are_nan_before_a_step_or_without_a_count():
    assert np.isnan(ts.TimeStepSize()(None, timer_context([])))
    untracked = timer_context([StepMetrics(step=1, step_size=2.0, newton_iterations=None)])
    assert ts.TimeStepSize()(None, untracked) == pytest.approx(2.0)
    assert np.isnan(ts.NewtonIterations()(None, untracked))


def test_timestep_vectors_need_the_timer_in_the_context():
    with pytest.raises(SummaryError, match="timer"):
        ts.TimeStepSize()(None, ScheduleContext(time=1.0))


def test_deck_requests_become_vectors(tmp_path):
    from bores.deck.file import DeckFile
    from bores.reservoir.deck import load_summary_quantities as load_reservoir
    from bores.timestep.deck import load_summary_quantities as load_timestep

    path = tmp_path / "case.DATA"
    path.write_text(
        "RUNSPEC\nDIMENS\n 2 2 1 /\nFIELD\nSUMMARY\nFOIP\nFPR\nRPR\n 1 /\nBPORV\n 2 1 1 /\n"
        "TIMESTEP\nNEWTON\nSCHEDULE\nTSTEP\n 10 /\n"
    )
    deck = DeckFile.from_path(path) if hasattr(DeckFile, "from_path") else DeckFile(path)

    assert {q.key for q in load_reservoir(deck)} == {"FPR", "FOIP", "RPR:1", "BPORV:2,1,1"}
    assert [q.key for q in load_timestep(deck)] == ["TIMESTEP", "NEWTON"]
