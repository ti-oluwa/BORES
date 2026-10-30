"""Builders for the states and reports the data frame tests convert."""

import datetime
import math

import numpy as np

from bores.grids.factories.cartesian import make_cartesian_grid
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.schedule.summary import SummaryReport
from bores.types import FluidPhase, PhaseValues, UnitSystem
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
)
from bores.wells.state import PerforationState, WellState, WellStates

GRID_SHAPE = (3, 2, 2)
N_CELLS = 12
START_DATE = datetime.datetime(2020, 1, 1)


def make_grid():
    nx, ny, nz = GRID_SHAPE
    return make_cartesian_grid(nx=nx, ny=ny, nz=nz, dx=10.0, dy=20.0, dz=5.0)


def make_reservoir_state(
    shift=0.0, unit_system=UnitSystem.FIELD, dtype=np.float64, hysteresis=False
):
    def filled(value=0.0):
        return np.full(N_CELLS, value, dtype=dtype)

    water = np.linspace(0.2, 0.4, N_CELLS).astype(dtype)
    gas = filled(0.05)
    state = ReservoirState(
        pressure=(3000.0 + np.arange(N_CELLS) + shift).astype(dtype),
        temperature=filled(180.0),
        oil_saturation=(1.0 - water - gas).astype(dtype),
        water_saturation=water,
        gas_saturation=gas,
        solution_gor=filled(500.0),
        oil_bubble_point_pressure=filled(2500.0),
        vaporized_oil_gas_ratio=filled(),
        gas_dew_point_pressure=filled(),
        gas_solubility_in_water=filled(),
        water_bubble_point_pressure=filled(),
        oil_mass=filled(1000.0),
        water_mass=filled(2000.0),
        free_gas_mass=filled(10.0),
        dissolved_gas_mass_in_oil=filled(5.0),
        dissolved_gas_mass_in_water=filled(),
        vaporized_oil_mass_in_gas=filled(),
        unit_system=unit_system,
    )
    if hysteresis:
        tracked = Hysteresis.from_saturation(
            water_saturation=water, gas_saturation=gas, dtype=dtype
        )
        state = state.evolve(hysteresis=tracked)
    return state


def make_perforation_state(cell_index, flowing_pressure, oil, water, gas):
    return PerforationState(
        perforation=Perforation(top_depth=1000.0, bottom_depth=1010.0),
        cell_index=cell_index,
        flowing_pressure=flowing_pressure,
        rates=PhaseValues(oil=oil, water=water, gas=gas),
    )


def make_well_states(shift=0.0):
    producer = WellState(
        well_name="PROD1",
        is_open=True,
        active_control=ProducerControl(
            mode=ProducerControlMode.OIL_RATE,
            target_rate=500.0,
            limits=(BHPLimit(min_value=1000.0),),
        ),
        bhp=2000.0 + shift,
        thp=300.0,
        perforation_states=(
            make_perforation_state(3, 2100.0 + shift, 100.0, 5.0, 50.0),
            make_perforation_state(9, 2200.0 + shift, 150.0, 6.0, 60.0),
        ),
        rates=PhaseValues(oil=250.0, water=11.0, gas=110.0),
        surface_rates=PhaseValues(oil=240.0, water=11.0, gas=100.0),
        active_limit=RateLimit(quantity=RateQuantity.LIQUID, max_value=900.0),
    )
    injector = WellState(
        well_name="INJ1",
        is_open=True,
        active_control=InjectorControl(
            injected_phase=FluidPhase.WATER,
            mode=InjectorControlMode.BHP,
            target_bhp=4500.0,
        ),
        bhp=4400.0,
        perforation_states=(make_perforation_state(1, 4300.0, 0.0, -800.0, 0.0),),
        rates=PhaseValues(oil=0.0, water=-800.0, gas=0.0),
        surface_rates=PhaseValues(oil=0.0, water=-790.0, gas=0.0),
    )
    shut = WellState(
        well_name="PROD2",
        is_open=False,
        active_control=ProducerControl(mode=ProducerControlMode.BHP, target_bhp=1500.0),
        bhp=1500.0,
        perforation_states=(),
        rates=PhaseValues(oil=0.0, water=0.0, gas=0.0),
        surface_rates=PhaseValues(oil=0.0, water=0.0, gas=0.0),
        active_limit=EconomicLimit(quantity=EconomicQuantity.WATER_CUT, max_value=0.9),
    )
    return WellStates({"PROD1": producer, "INJ1": injector, "PROD2": shut})


def make_summary_report():
    """FOPR and WBHP:PROD1 at times 1, 2, 3, WBHP:INJ1 at 1 and 3 only, and FOPT at 3 with no step or date."""
    report = SummaryReport()
    for step, time in enumerate([1.0, 2.0, 3.0]):
        date = START_DATE + datetime.timedelta(days=time)
        report.record(key="FOPR", time=time, value=100.0 - time, time_step=step, date=date)
        report.record(
            key="WBHP:PROD1", time=time, value=3000.0 - 10 * time, time_step=step, date=date
        )
        if step != 1:
            report.record(
                key="WBHP:INJ1", time=time, value=4000.0 + time, time_step=step, date=date
            )
    report.record(key="FOPT", time=3.0, value=5.0)
    return report


def is_missing(value):
    """Whether a value from either library stands for "no value": None, NaN, `pd.NA` or `NaT`."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return type(value).__name__ in {"NAType", "NaTType"}


def as_columns(frame):
    """
    Any pandas or polars frame as `{column: [values]}`, with every missing value as `None`.
    """
    if hasattr(frame, "to_pandas"):
        raw = frame.to_dict(as_series=False)
    else:
        raw = {name: frame[name].tolist() for name in frame.columns}
    return {
        name: [None if is_missing(value) else value for value in values]
        for name, values in raw.items()
    }


def is_categorical(series):
    """Whether a pandas or polars column is stored as a categorical."""
    return str(series.dtype).startswith(("category", "Categorical"))
