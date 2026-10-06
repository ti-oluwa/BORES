import pytest

from bores.serde.stores.json import JSONStore
from bores.types import FluidPhase, UnitSystem
from bores.wells.base import Perforation
from bores.wells.controls import (
    BHPLimit,
    EconomicLimit,
    EconomicQuantity,
    InjectorControl,
    InjectorControlMode,
    ProducerControl,
    ProducerControlMode,
    RateLimit,
    RateQuantity,
    THPLimit,
    WorkoverAction,
)
from bores.wells.state import PerforationState, PhaseValues, WellState

LIMITS = [
    RateLimit(quantity=RateQuantity.LIQUID, max_value=2000.0),
    BHPLimit(min_value=1000.0, max_value=4000.0),
    BHPLimit(min_value=900.0),
    THPLimit(max_value=800.0),
    EconomicLimit(
        quantity=EconomicQuantity.WATER_CUT,
        max_value=0.9,
        workover_action=WorkoverAction.PLUG,
        end_run=True,
    ),
    EconomicLimit(quantity=EconomicQuantity.OIL_RATE, min_value=10.0),
]

CONTROLS = [
    ProducerControl(mode=ProducerControlMode.OIL_RATE, target_rate=500.0, limits=tuple(LIMITS)),
    ProducerControl(mode=ProducerControlMode.BHP, target_bhp=1500.0),
    ProducerControl(mode=ProducerControlMode.THP, target_thp=300.0, vfp_table=2),
    InjectorControl(
        injected_phase=FluidPhase.WATER,
        mode=InjectorControlMode.RATE,
        target_rate=800.0,
        limits=(BHPLimit(max_value=5000.0),),
        guide_rate=2.0,
    ),
    InjectorControl(
        injected_phase=FluidPhase.GAS, mode=InjectorControlMode.BHP, target_bhp=4500.0
    ),
]


def make_state(control, active_limit):
    perforation_state = PerforationState(
        perforation=Perforation(top_depth=1000.0, bottom_depth=1010.0),
        cell_index=3,
        flowing_pressure=2500.0,
        phase_rates=PhaseValues(oil=1.0, water=0.5, gas=10.0),
    )
    return WellState(
        well_name="W1",
        is_open=True,
        active_control=control,
        bhp=2000.0,
        perforation_states=(perforation_state,),
        phase_rates=PhaseValues(oil=1.0, water=0.5, gas=10.0),
        surface_phase_rates=PhaseValues(oil=0.9, water=0.5, gas=9.0),
        active_limit=active_limit,
    )


@pytest.mark.parametrize("limit", LIMITS, ids=lambda limit: type(limit).__name__)
def test_limit_dump_and_load_keep_values(limit):
    loaded = type(limit).load(limit.dump())
    assert loaded == limit


@pytest.mark.parametrize("control", CONTROLS, ids=lambda control: type(control).__name__)
def test_control_dump_and_load_keep_values_and_limit_subclasses(control):
    loaded = type(control).load(control.dump())
    assert loaded == control
    assert tuple(type(limit) for limit in loaded.limits) == tuple(
        type(limit) for limit in control.limits
    )


@pytest.mark.parametrize("control", CONTROLS, ids=lambda control: type(control).__name__)
@pytest.mark.parametrize("active_limit", [None, BHPLimit(min_value=1000.0)], ids=["none", "bhp"])
def test_well_state_dump_contains_only_plain_data(control, active_limit):
    state = make_state(control, active_limit)
    dumped = state.dump()
    assert isinstance(dumped["active_control"], dict)
    assert dumped["active_limit"] is None or isinstance(dumped["active_limit"], dict)
    assert WellState.load(dumped) == state


@pytest.mark.parametrize("control", CONTROLS, ids=lambda control: type(control).__name__)
@pytest.mark.parametrize("limit", LIMITS, ids=lambda limit: type(limit).__name__)
def test_well_state_round_trips_through_a_json_file(tmp_path, control, limit):
    state = make_state(control, limit)
    path = tmp_path / "well_state.json"
    state.save(JSONStore(path))
    loaded = WellState.read(JSONStore(path))
    assert loaded == state
    assert type(loaded.active_control) is type(control)
    assert type(loaded.active_limit) is type(limit)


def test_well_state_round_trips_through_an_hdf5_file(tmp_path):
    pytest.importorskip("h5py")
    from bores.serde.stores.hdf5 import HDF5Store

    state = make_state(CONTROLS[0], LIMITS[4])
    path = tmp_path / "well_state.h5"
    state.save(HDF5Store(path))
    loaded = WellState.read(HDF5Store(path))
    assert loaded == state
    assert type(loaded.active_limit) is EconomicLimit


def test_unit_system_survives_the_round_trip():
    control = ProducerControl(
        mode=ProducerControlMode.BHP, target_bhp=100.0, unit_system=UnitSystem.METRIC
    )
    assert ProducerControl.load(control.dump()).unit_system is UnitSystem.METRIC
